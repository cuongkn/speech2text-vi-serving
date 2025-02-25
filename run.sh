#!/bin/bash

cd /speech2text-vietnamese-serving

python tllm/whisper/transformers_to_pt.py --model_name "$1" --dtype "$2"

bash tllm/whisper/build.sh "$1" "$2"

bash triton/whisper/prepare.sh "$1"
