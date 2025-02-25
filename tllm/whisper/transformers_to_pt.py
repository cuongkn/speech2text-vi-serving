import argparse
from copy import deepcopy
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq


WHISPER_MAPPING = {
    "layers": "blocks",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "layers": "blocks",
    ".self_attn.q_proj": ".attn.query",
    ".self_attn.k_proj": ".attn.key",
    ".self_attn.v_proj": ".attn.value",
    ".self_attn_layer_norm": ".attn_ln",
    ".self_attn.out_proj": ".attn.out",
    ".encoder_attn.q_proj": ".cross_attn.query",
    ".encoder_attn.k_proj": ".cross_attn.key",
    ".encoder_attn.v_proj": ".cross_attn.value",
    ".encoder_attn_layer_norm": ".cross_attn_ln",
    ".encoder_attn.out_proj": ".cross_attn.out",
    "decoder.layer_norm.": "decoder.ln.",
    "encoder.layer_norm.": "encoder.ln_post.",
    "embed_tokens": "token_embedding",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "layer_norm": "ln_post",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='whisper-medium-finetuned',)
    parser.add_argument('--output_dir',
                        type=str,
                        default='ckpt/pt_checkpoint',
                        help='The path to save the .pt checkpoint')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()
    return args


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def convert_hf_ckpt_to_whisper_ckpt(hf_model_name_or_path, whisper_ckpt_save_path, dtype):
    pretrained_path = os.path.join('ckpt/pretrained_checkpoint', hf_model_name_or_path)
    transformer_model = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_path)
    
    if dtype != "float32":
        transformer_model = transformer_model.half()
    config = transformer_model.config

    dims = {
        'n_mels': config.num_mel_bins,
        'n_vocab': config.vocab_size,
        'n_audio_ctx': config.max_source_positions,
        'n_audio_state': config.d_model,
        'n_audio_head': config.encoder_attention_heads,
        'n_audio_layer': config.encoder_layers,
        'n_text_ctx': config.max_target_positions,
        'n_text_state': config.d_model,
        'n_text_head': config.decoder_attention_heads,
        'n_text_layer': config.decoder_layers
    }

    state_dict = deepcopy(transformer_model.model.state_dict())
    state_dict = rename_keys(state_dict)

    torch.save({"dims": dims, "model_state_dict": state_dict}, whisper_ckpt_save_path)


if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    pt_saving_path = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}.pt")
    
    print(f"Convering hf model {args.model_name} to .pt {args.dtype} format")
    
    convert_hf_ckpt_to_whisper_ckpt(args.model_name, pt_saving_path, args.dtype)
    
    