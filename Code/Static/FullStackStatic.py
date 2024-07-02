from ASR import ASR_audio_file
from LLM import LLM_response
from TTS_file import TTS_Mozilla

# Acquire test Audio
path_audio = "Neuphonic/Data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"

# Convert it to Text
text = ASR_audio_file(audio_file_path=path_audio)
print(f"Provided speech to text: {text}")

# Generate response via LLM
response = LLM_response(text_prompt=text)
print(f"Provided LLM response: {response}")

# Convert response to speech and save it to desired path
audio_name = "84-121123-0000"
TTS_Mozilla(response, audio_name)
print("Response saved")