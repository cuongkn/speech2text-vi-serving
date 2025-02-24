ARG BASE_IMAGE

FROM ${BASE_IMAGE}

RUN apt update && apt-get install -y ffmpeg

COPY triton/whisper /triton/whisper

RUN python3 -m pip uninstall tensorrt -y
RUN python3 -m pip install -r /triton/whisper/requirements.txt

ENTRYPOINT ["/bin/bash"]
