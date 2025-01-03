import logging
from contextlib import AbstractContextManager, nullcontext

import torch
from torch.distributed import ReduceOp
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoConfig

from .containers import LRSchedulerContainer, ModelContainer, OptimizerContainer
from .data import ResumableDataLoader, get_next_batch
from .distributed import dtensor_to_tensor
from .enums import GradientCheckpointingMethod
from .hf_models import is_custom_model
from .hf_models.modeling_utils import is_glu
from .model_wrapper import ModelWrapper
from .utils import ExperimentsTracker, MetricsTrackingDict, ProcessGroupManager, is_torchao_available, log_metrics


if is_torchao_available():
    from .distributed import FP8Manager


def train_step(
    model_container: ModelContainer,
    pipeline_schedule: _PipelineSchedule,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: ResumableDataLoader,
    gradient_accumulation_steps: int,
    gradient_clipping: float,
    forward_context: AbstractContextManager,
    backward_context: AbstractContextManager,
    sync_every_gradient_accumulation_step: bool,
    is_pipeline_parallel_enabled: bool,
    local_batch_size: int,
    micro_batch_size: int,
    sequence_length: int,
) -> MetricsTrackingDict:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model_container (ModelContainer): container of models
        pipeline_schedule (_PipelineSchedule): pipeline schedule
        optimizer_container (OptimizerContainer): container of optimizers
        lr_scheduler_container (LRSchedulerContainer): container of learning rate schedulers
        train_dataloader (ResumableDataLoader): training dataloader
        gradient_accumulation_steps (int): gradient accumulation steps
        gradient_clipping (float): gradient clipping value
        forward_context (AbstractContextManager): a context that is used for every model forward call
        backward_context (AbstractContextManager): a context that is used for every model backward call
        sync_every_gradient_accumulation_step (bool): whether to sync on every gradient accumulation step
        is_pipeline_parallel_enabled (bool): whether to use pipeline parallel
        local_batch_size (int): local batch size
        sequence_length (int): sequence length

    Returns:
        MetricsTrackingDict: metrics to track
    """

    assert len(model_container) == len(optimizer_container)
    assert len(optimizer_container) == len(lr_scheduler_container)

    if is_pipeline_parallel_enabled:
        metrics_tracker = _train_step_with_pipeline_parallel(
            model_container=model_container,
            pipeline_schedule=pipeline_schedule,
            optimizer_container=optimizer_container,
            lr_scheduler_container=lr_scheduler_container,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            local_batch_size=local_batch_size,
            sequence_length=sequence_length,
        )
    else:
        assert len(model_container) == 1

        metrics_tracker = _train_step_without_pipeline_parallel(
            model=model_container[0],
            optimizer=optimizer_container[0],
            lr_scheduler=lr_scheduler_container[0],
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            forward_context=forward_context,
            backward_context=backward_context,
            sync_every_gradient_accumulation_step=sync_every_gradient_accumulation_step,
            micro_batch_size=micro_batch_size,
            sequence_length=sequence_length,
        )

    return metrics_tracker


def _train_step_with_pipeline_parallel(
    model_container: ModelContainer,
    pipeline_schedule: _PipelineSchedule,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: ResumableDataLoader,
    gradient_accumulation_steps: int,
    gradient_clipping: float,
    local_batch_size: int,
    sequence_length: int,
) -> MetricsTrackingDict:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model_container (ModelContainer): container of models
        pipeline_schedule (_PipelineSchedule): pipeline schedule
        optimizer_container (OptimizerContainer): container of optimizers
        lr_scheduler_container (LRSchedulerContainer): container of learning rate schedulers
        train_dataloader (ResumableDataLoader): training dataloader
        gradient_accumulation_steps (int): gradient accumulation steps
        gradient_clipping (float): gradient clipping value
        local_batch_size (int): local batch size
        sequence_length (int): sequence length

    Returns:
        MetricsTrackingDict: metrics to track
    """

    fsdp_algorithm = 2 if hasattr(model_container[0], "set_requires_gradient_sync") else 1
    grad_norm = []

    optimizer_container.zero_grad()

    batch = get_next_batch(train_dataloader)

    if ProcessGroupManager.is_tensor_parallel_first_rank():
        batch = batch["text"]

    batch = model_container[0].broadcast_tensor_parallel_input(batch, (local_batch_size, sequence_length + 1))

    is_first_pipeline_rank = ProcessGroupManager.get_pipeline_parallel_rank() == 0
    is_last_pipeline_rank = (
        ProcessGroupManager.get_pipeline_parallel_rank() == ProcessGroupManager.get_pipeline_parallel_world_size() - 1
    )

    if is_first_pipeline_rank:
        pipeline_schedule.step(batch)
    elif is_last_pipeline_rank:
        losses = []
        labels = batch[:, 1:]
        pipeline_schedule.step(target=labels, losses=losses)
    else:
        pipeline_schedule.step()

    if gradient_clipping is not None:
        for model in model_container:
            if fsdp_algorithm == 1:
                grad_norm.append(model.clip_grad_norm_(gradient_clipping))
            else:
                grad_norm.append(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping))

    if is_torchao_available():
        FP8Manager.sync_float8_amax_and_scale_history(model_container)

    optimizer_container.step()
    lr_scheduler_container.step()

    if is_torchao_available():
        FP8Manager.precompute_float8_dynamic_scale_for_fsdp(model_container)

    metrics_tracker = MetricsTrackingDict({})

    with torch.inference_mode():
        grad_norm = dtensor_to_tensor(sum(grad_norm))
        torch.distributed.all_reduce(grad_norm, group=ProcessGroupManager.get_pipeline_parallel_group())

        if is_last_pipeline_rank:
            losses = sum(losses)

            metrics_tracker = metrics_tracker + {"loss": losses, "grad_norm": grad_norm}
            metrics_tracker = metrics_tracker + model.get_extra_metrics()
            model.reset_extra_metrics()

            metrics_tracker = metrics_tracker / gradient_accumulation_steps

            metrics_tracker["grad_norm"] = grad_norm

            for key in metrics_tracker:
                metrics_tracker[key] = dtensor_to_tensor(metrics_tracker[key])

            metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

    return metrics_tracker


def _train_step_without_pipeline_parallel(
    model: ModelWrapper,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: ResumableDataLoader,
    gradient_accumulation_steps: int,
    gradient_clipping: float,
    forward_context: AbstractContextManager,
    backward_context: AbstractContextManager,
    sync_every_gradient_accumulation_step: bool,
    micro_batch_size: int,
    sequence_length: int,
) -> MetricsTrackingDict:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model (ModelWrapper): model
        optimizer (Optimizer): optimizer
        lr_scheduler (LamdaLR): learning rate scheduler
        train_dataloader (ResumableDataLoader): training dataloader
        gradient_accumulation_steps (int): gradient accumulation steps
        gradient_clipping (float): gradient clipping value
        forward_context (AbstractContextManager): a context that is used for every model forward call
        backward_context (AbstractContextManager): a context that is used for every model backward call
        sync_every_gradient_accumulation_step (bool): whether to sync on every gradient accumulation step
        micro_batch_size (int): micro batch size
        sequence_length (int): sequence length

    Returns:
        MetricsTrackingDict: metrics to track
    """

    fsdp_algorithm = 2 if hasattr(model, "set_requires_gradient_sync") else 1

    no_sync = nullcontext
    if not sync_every_gradient_accumulation_step:
        if fsdp_algorithm == 1:
            no_sync = model.no_sync
        else:
            model.set_requires_gradient_sync(False)

    metrics_tracker = MetricsTrackingDict({})
    grad_norm = None
    optimizer.zero_grad()

    lm_loss_multiplier = 1 / (micro_batch_size * sequence_length)

    with no_sync():
        for _ in range(gradient_accumulation_steps - 1):
            batch = get_next_batch(train_dataloader)
            with forward_context():
                loss_micro_step_dict = model(batch, lm_loss_multiplier=lm_loss_multiplier)

            # compute gradients
            with backward_context():
                loss_micro_step_scaled: torch.Tensor = loss_micro_step_dict["loss"] / gradient_accumulation_steps
                loss_micro_step_scaled.backward()

            with torch.inference_mode():
                metrics_tracker = metrics_tracker + loss_micro_step_dict

    if fsdp_algorithm == 2:
        model.set_requires_gradient_sync(True)

    batch = get_next_batch(train_dataloader)
    with forward_context():
        loss_micro_step_dict = model(batch, lm_loss_multiplier=lm_loss_multiplier)

    # compute gradients
    with backward_context():
        loss_micro_step_scaled: torch.Tensor = loss_micro_step_dict["loss"] / gradient_accumulation_steps
        loss_micro_step_scaled.backward()

    with torch.inference_mode():
        metrics_tracker = metrics_tracker + loss_micro_step_dict

    if gradient_clipping is not None:
        if fsdp_algorithm == 1:
            grad_norm = model.clip_grad_norm_(gradient_clipping)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

    if is_torchao_available():
        FP8Manager.sync_float8_amax_and_scale_history([model])

    optimizer.step()
    lr_scheduler.step()

    if is_torchao_available():
        FP8Manager.precompute_float8_dynamic_scale_for_fsdp([model])

    with torch.inference_mode():
        metrics_tracker = metrics_tracker / gradient_accumulation_steps

        metrics_tracker["grad_norm"] = (
            torch.tensor(0, device=torch.cuda.current_device()) if grad_norm is None else grad_norm
        )

        for key in metrics_tracker:
            metrics_tracker[key] = dtensor_to_tensor(metrics_tracker[key])

        metrics_tracker = all_reduce_metrics_tracker(metrics_tracker)

    return metrics_tracker


def all_reduce_metrics_tracker(metrics_tracker: MetricsTrackingDict) -> MetricsTrackingDict:
    tensor = [metrics_tracker[key] for key in metrics_tracker]
    tensor = torch.stack(tensor)
    # NOTE the cpu() call was to save memory but might not be needed anymore
    # tensor = torch.stack(tensor) / ProcessGroupManager.get_data_parallel_world_size()
    # tensor = tensor.cpu()
    # gloo op doesn't support averaging so we do sum and divide by world size above
    torch.distributed.all_reduce(tensor, op=ReduceOp.AVG, group=ProcessGroupManager.get_data_parallel_group())
    tensor = tensor.tolist()

    for i, key in enumerate(metrics_tracker):
        metrics_tracker[key] = tensor[i]

    return metrics_tracker


def track_metrics(
    global_step: int, experiments_tracker: ExperimentsTracker, metrics_tracker: MetricsTrackingDict, context: str
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        metrics_tracker (float): metrics tracker
        context (str): experiment context
    """

    # experiments tracker
    experiments_tracker.track(metrics_tracker.get_dict(), step=global_step, context=context)

    message = f"step = {global_step}"
    for key in metrics_tracker:
        if key == "learning_rate":
            message += f", {key} = {metrics_tracker[key]:.4e}"
        else:
            message += f", {context}-{key} = {metrics_tracker[key]:.4f}"

    log_metrics(logging.INFO, message)


def get_torch_profiler(torch_profiler_trace_path: str) -> torch.profiler.profile:
    torch_profiler = None
    if torch_profiler_trace_path is not None:
        torch_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=5 if ProcessGroupManager.get_global_rank() == 0 else 150000, warmup=5, active=1, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(torch_profiler_trace_path),
            record_shapes=True,
        )

    return torch_profiler


def get_model_tflops(
    config: AutoConfig,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing_method: GradientCheckpointingMethod | None,
    gradient_checkpointing_args: dict,
) -> None:
    if not is_custom_model(config.model_type):
        return 0

    b = batch_size
    s = sequence_length
    h = config.n_embd
    f = config.n_inner
    n = config.n_head
    k = config.num_key_value_heads
    l = config.n_layer
    v = config.vocab_size

    mlp_flops = 4 * b * s * h * f
    if config.model_type == "moe_dolomite":
        mlp_flops *= config.num_experts_per_tok

    if is_glu(config.activation_function):
        mlp_flops *= 1.5

    attention_flops = 4 * b * s * h * (h * (1 + k / n) + s)

    forward_flops = attention_flops + mlp_flops

    if gradient_checkpointing_method == GradientCheckpointingMethod.block:
        num_layers_checkpointed = gradient_checkpointing_args.get("num_blocks", l)
        fraction_of_layers_checkpointed = num_layers_checkpointed / l
        backward_flops = (2 + fraction_of_layers_checkpointed) * forward_flops
    else:
        backward_flops = 2 * forward_flops

    model_flops = l * (forward_flops + backward_flops)
    model_flops += 6 * b * s * h * v
    model_flops /= 10**12

    return model_flops
