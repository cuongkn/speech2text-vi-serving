FROM nvcr.io/nvidia/tritonserver:24.05-py3
LABEL maintainer="NVIDIA"
LABEL repository="tritonserver"

RUN apt update && apt-get install -y ffmpeg
# RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
RUN pip3 install tensorrt_llm==0.14.0.dev2024091000 -U --extra-index-url https://pypi.nvidia.com
# RUN python3 -m pip install tensorrt_llm==0.12.0
# RUN python3 -m pip install tritonclient[all]

COPY triton/whisper /workspace/triton/whisper
COPY assets/whisper /workspace/assets/whisper 

WORKDIR /workspace

# RUN python3 -m pip install -r triton/whisper/requirements.txt

ENTRYPOINT ["/bin/bash"]