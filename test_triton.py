import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import soundfile


def load_audio(wav_path):
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    return waveform, sample_rate

def send_whisper(
    client,
    audio_filepath: str,
    model_name: str,
    padding_duration: int = 10,
    whisper_prompt: str = "<|startoftranscript|><|vi|><|transcribe|><|notimestamps|>",
):
    waveform, sample_rate = load_audio(audio_filepath)
    duration = int(len(waveform) / sample_rate)

    # padding to nearset 10 seconds
    samples = np.zeros(
        (
            1,
            padding_duration * sample_rate * ((duration // padding_duration) + 1),
        ),
        dtype=np.float32,
    )

    samples[0, : len(waveform)] = waveform

    lengths = np.array([[len(waveform)]], dtype=np.int32)
    
    print(type(samples))

    inputs = [
        grpcclient.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        grpcclient.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)

    input_data_numpy = np.array([whisper_prompt], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[1].set_data_from_numpy(input_data_numpy)

    outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTS")]

    response = client.infer(
        model_name=model_name, inputs=inputs, outputs=outputs
    )
    
    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        
    decoding_results = decoding_results.decode("utf-8")
    
    return decoding_results


if __name__=="__main__":
    triton_url="0.0.0.0:8001"
    client = grpcclient.InferenceServerClient(url=triton_url)
    transcript = send_whisper(
        client,
        audio_filepath="examples/VIVOSDEV01_R002.wav",
        model_name="phowhisper_medium_finetuned"
    )
    print(transcript)
    