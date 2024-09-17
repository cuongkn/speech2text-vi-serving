export CUDA_VISIBLE_DEVICES="0"

model_repo_path=model_repo_whisper_tllm

tritonserver --model-repository $model_repo_path \
            --pinned-memory-pool-byte-size=4096000000 \
            --cuda-memory-pool-byte-size=0:8192000000 \
            --http-port 10086 \
            --metrics-port 10087