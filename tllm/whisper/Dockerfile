FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs
RUN pip3 install tensorrt-llm --no-cache-dir -U --extra-index-url https://pypi.nvidia.com

COPY tllm/whisper-tensorrt-llm /workspace/tllm/whisper-tensorrt-llm
COPY assets/whisper workspace/assets/whisper 

WORKDIR /workspace 

RUN pip install -r tllm/whisper-tensorrt-llm/requirements.txt

ENTRYPOINT ["/bin/bash"]

CMD ["-c", "python3 -c 'import tensorrt_llm'"]