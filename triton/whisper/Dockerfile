FROM nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3

RUN apt update && apt-get install -y ffmpeg

COPY triton/whisper /triton/whisper
COPY engines /engines

RUN python3 -m pip uninstall tensorrt -y
RUN python3 -m pip install -r /triton/whisper/requirements.txt

ENTRYPOINT ["/bin/bash"]
