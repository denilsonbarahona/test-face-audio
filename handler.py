##from transformers import pipeline
##import scipy
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import runpod
import torch
import soundfile as sf
from transformers import pipeline, MusicgenForConditionalGeneration, AutoProcessor
##import scipy
## from huggingface_hub import hf_hub_download, snapshot_download
from utils.storage import storage_client
from utils.config import config

runpod.api_key = config.get_runpod_api_key()


synthesiser = pipeline(
    "text-to-audio",
    "facebook/musicgen-stereo-large",
    device="cuda:0" if torch.cuda.is_available() else "cpu",  # Usa GPU si está disponible
    torch_dtype=torch.float16  # Precisión mixta para mayor eficiencia en GPU
)


#https://huggingface.co/FunAudioLLM/InspireMusic-1.5B-Long.git

async def handler(event):
    print('li')
    try:

        input_data = event["input"]
        prompt = input_data["prompt"]
        fileName = input_data["fileName"]
        duration = input_data["duration"]
        music = synthesiser(
            prompt,
            forward_params={"max_new_tokens": duration}  # 120 segundos
        )
        sf.write(fileName, music["audio"][0].T, music["sampling_rate"])
        file_path = os.path.join(os.getcwd(), fileName)
        audio_url = storage_client.upload_file(file_path, fileName)
        response = {"url": audio_url}
        return response
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})



   # scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
    # sf.write("musicgen_2min.wav", music["audio"][0].T, music["sampling_rate"])
    ##model_path = os.path.abspath("./InspireMusic-1.5B-Long")
    ## snapshot_download(repo_id="FunAudioLLM/InspireMusic-1.5B-Long", local_dir=model_path)
    #try:
    #    music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})

    #    scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
    #    file_path = os.path.join(os.getcwd(), "musicgen_out.wav")
    #    audio_url = storage_client.upload_file(file_path, "musicgen_out.wav")
    #    response = {"url": audio_url}
    #    return response
    # except Exception as e:
    #    return {"error": str(e)}
