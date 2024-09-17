model_name="$1"

INFERENCE_PRECISION=float16
WEIGHT_ONLY_PRECISION=int8
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8
checkpoint_path=engines/
checkpoint_dir=${checkpoint_path}${model_name}_weights
output_dir=${checkpoint_path}${model_name}

python3 tllm/whisper-tensorrt-llm/convert_checkpoint.py \
                --model_name $model_name \
                --dtype float16 \
                --logits_dtype float16 \
                --output_dir $checkpoint_dir

# Build the large-v3 model using trtllm-build
trtllm-build  --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --paged_kv_cache disable \
              --remove_input_padding disable \
              --context_fmha disable \
              --moe_plugin disable \
              --enable_xqa disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --remove_input_padding disable \
              --max_input_len 3000 \
              --max_seq_len=3000

trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --paged_kv_cache disable \
              --remove_input_padding disable \
              --context_fmha disable \
              --moe_plugin disable \
              --enable_xqa disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}