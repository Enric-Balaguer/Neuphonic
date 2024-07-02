import whisper
import pyaudio
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
import soundfile as sf
import os

# Ensure the output directory exists
os.makedirs("Neuphonic/Responses", exist_ok=True)

# Load all models first
ASR_model = whisper.load_model("base")

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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") # Use own token
cache_dir = 'Neuphonic/Models'
LLM_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")

manager = ModelManager(output_prefix="Neuphonic/Models")
model_name = "tts_models/en/ljspeech/vits"
model_path, config_path, model_item = manager.download_model(model_name)
TTS_synthesizer = Synthesizer(
    model_path,
    config_path,
    use_cuda=True  # Set to False if you want to use CPU
)

# Start listening to audio from your mic.

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define stream parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1024
BUFFER_DURATION = 5  # Duration in seconds to buffer audio
BUFFER_SIZE = int(RATE / CHUNK_SIZE * BUFFER_DURATION)

# Setup audio input and output streams
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

output_stream = p.open(format=pyaudio.paFloat32,
                       channels=1,
                       rate=22050,
                       output=True)

print("Listening... Press Ctrl+C to stop.")

try:
    audio_buffer = np.empty(0, dtype=np.float32)
    
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.float32)
        audio_buffer = np.append(audio_buffer, audio_data)

        # Check if buffer has enough data to process
        if len(audio_buffer) >= BUFFER_SIZE:
            # Process audio
            result = ASR_model.transcribe(audio_buffer, temperature=0)
            # Clear the buffer or handle as needed
            audio_buffer = np.empty(0, dtype=np.float32)

            # Tokenize the input
            inputs = tokenizer(result["text"], return_tensors="pt").to(LLM_model.device)

            # Generate output
            outputs = LLM_model.generate(
                **inputs,
                max_new_tokens=50,  # Allowing for more extended output
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode the generated tokens, ensuring to skip the tokens used for the prompt
            generated_tokens = outputs[:, inputs['input_ids'].size(1):][0]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Generate speech
            wav = TTS_synthesizer.tts(result.strip())

            # Ensure wav is a NumPy array
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav)
            
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)

            # Normalize the audio to be between -1 and 1
            wav = np.clip(wav, -1, 1)

            # Play the audio
            output_stream.write(wav.tobytes())

except KeyboardInterrupt:
    print("Stopping...")

# Clean up
finally:
    stream.stop_stream()
    stream.close()
    output_stream.stop_stream()
    output_stream.close()
    p.terminate()