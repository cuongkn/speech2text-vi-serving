FROM nvcr.io/nvidia/tritonserver:24.05-py3

RUN apt update && apt-get install -y ffmpeg
RUN pip3 install tensorrt-llm==0.11.0.dev2024052800

COPY triton/whisper /triton/whisper
COPY assets/whisper /assets/whisper 
COPY engines /engines

RUN python3 -m pip install -r /triton/whisper/requirements.txt

ENTRYPOINT ["/bin/bash"]