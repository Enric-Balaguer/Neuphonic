from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
import soundfile as sf
import os
import numpy as np
import pyaudio

def TTS_Mozilla(input_text: str, output_audio_name: str):
    """
    Using Mozilla TTS, convert input text to audio file
    """
    # Ensure the output directory exists
    os.makedirs("Neuphonic/Responses", exist_ok=True)

    # Initialize ModelManager
    manager = ModelManager(output_prefix="Neuphonic/Models")

    model_name = "tts_models/en/ljspeech/vits"
    model_path, config_path, model_item = manager.download_model(model_name)

    # Initialize the Synthesizer
    synthesizer = Synthesizer(
        model_path,
        config_path,
        use_cuda=True  # Set to False if you want to use CPU
    )

    # Generate speech
    wav = synthesizer.tts(input_text)

    # Ensure wav is a NumPy array
    if not isinstance(wav, np.ndarray):
        wav = np.array(wav)
    
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    # Normalize the audio to be between -1 and 1
    wav = np.clip(wav, -1, 1)

    # Save the audio to a file
    output_path = "Neuphonic/Responses/" + output_audio_name + ".wav"
    sf.write(output_path, wav, 22050)

    print(f"Speech generated and saved to {output_path}")