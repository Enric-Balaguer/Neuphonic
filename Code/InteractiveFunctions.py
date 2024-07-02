from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
import soundfile as sf
import os
import numpy as np
import whisper
import pyaudio
import threading
import soundfile as sf

# For the interactive session, I want all models to be loaded first

# Global variables for models
llm_model = None
llm_tokenizer = None
asr_model = None
tts_synthesizer = None

def load_models():
    global llm_model, llm_tokenizer, asr_model, tts_synthesizer

    # Load LLM (Mistral 7B)
    model_name = "mistralai/Mistral-7B-v0.1"
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
    cache_dir = 'Neuphonic/Models'
    llm_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")

    # Load ASR model (whisper from OpenAI)
    asr_model = whisper.load_model("base")

    # Load TTS synthesiser 
    manager = ModelManager(output_prefix="Neuphonic/Models")

    model_name = "tts_models/en/ljspeech/vits"
    model_path, config_path, model_item = manager.download_model(model_name)

    # Initialize the Synthesizer
    tts_synthesizer = Synthesizer(
        model_path,
        config_path,
        use_cuda=True  # Set to False if you want to use CPU
    )

    print("All models loaded successfully")

def LLM_response(text_prompt:str):
    """
    Using Mistral 7B model, generate response from provided text_prompt.
    text_prompt: string prompt for Mistral 7B model.
    """
    global llm_model, llm_tokenizer

    if llm_model is None or llm_tokenizer is None:
        raise RuntimeError("LLM model and tokenizer must be loaded before calling LLM_response")
    
    # Tokenize the input
    inputs = llm_tokenizer(text_prompt, return_tensors="pt").to(llm_model.device)

    # Generate output
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=50,  # Allowing for more extended output
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        pad_token_id=llm_tokenizer.eos_token_id,
    )

    # Decode the generated tokens, ensuring to skip the tokens used for the prompt
    generated_tokens = outputs[:, inputs['input_ids'].size(1):][0]
    result = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return result.strip()

def ASR_audio_file(audio_file_path:str):
    """
    Converts audio to text
    audio_file_path: string containing the path of the audio to be converted to text.
    """
    global asr_model

    if asr_model is None:
        raise RuntimeError("ASR model must be loaded before calling ASR_audio_file")

    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(asr_model.device)

    # Detect the spoken language
    _, probs = asr_model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(asr_model, mel, options)

    return result.text

def TTS_Mozilla(input_text: str, output_path: str):
    """
    Using Mozilla TTS, convert input text to audio file
    """
    global tts_synthesizer

    if tts_synthesizer is None:
        raise RuntimeError("TTS Synthesizer must be loaded before calling TTS_Mozilla")

    # Generate speech
    wav = tts_synthesizer.tts(input_text)

    # Ensure wav is a NumPy array
    if not isinstance(wav, np.ndarray):
        wav = np.array(wav)
    
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    # Normalize the audio to be between -1 and 1
    wav = np.clip(wav, -1, 1)

    # Save the audio to a file
    sf.write(output_path, wav, 22050)

    print(f"Speech generated and saved to {output_path}")

    # Stream the audio
    def stream_audio():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=22050,
                        output=True)
        
        # Convert to bytes
        audio_bytes = wav.tobytes()
        
        # Stream in chunks
        chunk_size = 1024
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            stream.write(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()

    # Start streaming in a separate thread
    thread = threading.Thread(target=stream_audio)
    thread.start()

    return output_path, thread