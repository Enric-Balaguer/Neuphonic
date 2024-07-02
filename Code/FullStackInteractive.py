from ListenForAudio import record_audio
from InteractiveFunctions import load_models, LLM_response, ASR_audio_file, TTS_Mozilla

# Load all models first
load_models()

# Start "interactive" session
recording_path = "Neuphonic/Data/Recordings/recorded_audio.wav"
output_audio_path = "Neuphonic/Data/Responses/response_audio.wav"

while True:

    record_audio(recording_path)

    transcribed_text = ASR_audio_file(recording_path)
    print(f"Transcribed text is: {transcribed_text}")

    response_text = LLM_response(transcribed_text)
    print(f"LLM response is: {response_text}")

    _, audio_thread = TTS_Mozilla(response_text, output_audio_path)
    print("Audio started playing")
    audio_thread.join() #Dangerous to block application until audio finished but for demonstrative purposes only