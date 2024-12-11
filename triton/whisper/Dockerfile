FROM nvcr.io/nvidia/tritonserver:24.05-py3

RUN apt update && apt-get install -y ffmpeg
RUN pip3 install tensorrt-llm==0.11.0.dev2024052800 
RUN pip3 install pynvml==11.5.3 tensorrt-cu12==10.5.0

COPY triton/whisper /triton/whisper
COPY engines /engines

RUN python3 -m pip install -r /triton/whisper/requirements.txt

ENTRYPOINT ["/bin/bash"]