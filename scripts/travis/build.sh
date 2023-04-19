REGISTRY_URL=us.icr.io
NAMESPACE=cil15-shared-registry
ARCH=amd64

if [ $BUILD_IMAGE = "true" ]
then
    # login into cil15-registry
    echo $IBM_CLOUD_API_KEY | docker login -u iamapikey --password-stdin $REGISTRY_URL

    if [ $TRAVIS_PULL_REQUEST != "false" ]
    then
        IMAGE=$REGISTRY_URL/$NAMESPACE/$IMAGE_NAME:$IMAGE_TAG-$TRAVIS_JOB_ID
    else
        IMAGE=$REGISTRY_URL/$NAMESPACE/$IMAGE_NAME:$IMAGE_TAG
    fi

    echo "building image $IMAGE"

    docker build -t $IMAGE --platform $ARCH .
    docker push $IMAGE
fi