launch_triton_repo_python_backend() {

    local engine_dir=$1

    n_mels=$(cat ${engine_dir}/encoder/config.json | grep n_mels | awk -F': ' '{print $2}' | tr -d ',')

    if [[ "$engine_dir" == *"multi-hans"* ]]; then
        zero_pad=true # fine-tuned model could remove 30s padding, so set pad to none
    else
        zero_pad=false
    fi

    echo "engine_dir: $engine_dir", "n_mels: $n_mels", "zero_pad: $zero_pad"

    model_repository=whisper_model_repository
    
    rm -rf $model_repository
    cp -r /triton/whisper/whisper_model_repository_trtllm $model_repository
    wget -nc --directory-prefix=$model_repository/infer_bls/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
    wget -nc --directory-prefix=$model_repository/whisper/1 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

    TRITON_MAX_BATCH_SIZE=64
    MAX_QUEUE_DELAY_MICROSECONDS=100

    python3 /triton/whisper/fill_template.py -i $model_repository/whisper/config.pbtxt engine_dir:${engine_dir},n_mels:$n_mels,zero_pad:$zero_pad,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 /triton/whisper/fill_template.py -i $model_repository/infer_bls/config.pbtxt engine_dir:${engine_dir},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    
    echo "Launching triton server with model_repo: $model_repository"
    
    tritonserver --model-repository=$model_repository
}


MODEL_IDs=("whisper-medium-finetuned")
CUDA_VISIBLE_DEVICES=0

model_id=$1

engine_dir="/engines/${model_id}"

if printf '%s\n' "${MODEL_IDs[@]}" | grep -q "^$model_id$"; then
    launch_triton_repo_python_backend "$engine_dir" || exit 1
else
    echo "$model_id is NOT in the MODEL_IDs array."
    exit 1
fi
