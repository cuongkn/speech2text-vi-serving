
#!/bin/bash

# Check if a version was passed as an argument
if [ -z "$1" ]; then
    echo "Error: Please provide the tensorrt-llm version as an argument."
    echo "Usage: ./build_image.sh <tensorrt-llm-version>"
    exit 1
fi

# The version of tensorrt-llm passed as the first argument
TENSORRT_LLM_VERSION=$1

echo "tensorrt-llm==$TENSORRT_LLM_VERSION" >> triton/whisper/requirements.txt
echo "Appended tensorrt-llm==$TENSORRT_LLM_VERSION to requirements.txt."

# Define the mapping of tensorrt-llm versions to base image tags
declare -A VERSION_TO_BASE_IMAGE
VERSION_TO_BASE_IMAGE=(
    [0.15.0]=nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3
    [0.14.0]=nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3
    [0.13.0]=nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3
    [0.12.0]=nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
    [0.11.0]=nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3
    [0.18.0.dev2025021800]=nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3
)

# Get the corresponding base image for the tensorrt-llm version
BASE_IMAGE=${VERSION_TO_BASE_IMAGE[$TENSORRT_LLM_VERSION]}

if [ -z "$BASE_IMAGE" ]; then
    echo "Error: No base image found for tensorrt-llm version $TENSORRT_LLM_VERSION"
    exit 1
fi

if [ "$BASE_IMAGE" == "nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3" ]; then
    echo "torch==2.6.0" >> triton/whisper/requirements.txt
    echo "torchvision==0.21.0" >> triton/whisper/requirements.txt
fi

BASE_IMAGE_SUFFIX=$(echo $BASE_IMAGE | sed 's|^nvcr.io/nvidia/||' | cut -d':' -f2-)
CUSTOM_TAG="cuongkn/triton-whisper-$1:$BASE_IMAGE_SUFFIX-0.0.1"
echo "The image will be tagged as: $CUSTOM_TAG"

docker build --rm -t $CUSTOM_TAG -f triton/whisper/Dockerfile --build-arg BASE_IMAGE=$BASE_IMAGE .

if [ "$BASE_IMAGE" == "nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3" ]; then
    sed -i -e '$d' -e '$d' triton/whisper/requirements.txt
fi

sed -i -e '$d' triton/whisper/requirements.txt