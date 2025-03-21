##from transformers import pipeline
##import scipy
import os
import runpod
from transformers import pipeline, MusicgenForConditionalGeneration, AutoProcessor
import scipy
from utils.storage import storage_client
from utils.config import config

runpod.api_key = config.get_runpod_api_key()


synthesiser = pipeline("text-to-audio", "facebook/musicgen-large")


async def handler(event):
    try:
        music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})

        scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
        file_path = os.path.join(os.getcwd(), "musicgen_out.wav")
        audio_url = storage_client.upload_file(file_path, "musicgen_out.wav")
        response = {"url": audio_url}
        return response
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})

