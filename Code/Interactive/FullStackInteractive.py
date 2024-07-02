from ListenForAudio import record_audio
from InteractiveFunctions import load_models, LLM_response, ASR_audio_file, TTS_Mozilla

def interactive_session():
    # Load all models first
    print("Loading models...")
    load_models()
    print("Models loaded successfully.")

    # Start "interactive" session
    recording_path = "Neuphonic/Data/Recordings/recorded_audio.wav"
    output_audio_path = "Neuphonic/Data/Responses/response_audio.wav"

    print("\nWelcome to the interactive AI assistant!")
    print("You can speak, and the AI will respond.")
    print("Type 'exit' at any time to end the session.")

    while True:
        user_input = input("\nPress Enter to start recording, or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            print("Exiting the session. Goodbye!")
            break

        record_audio(recording_path)

        print("Transcribing your audio...")
        transcribed_text = ASR_audio_file(recording_path)
        print(f"Transcribed text: {transcribed_text}")

        print("Generating AI response...")
        response_text = LLM_response(transcribed_text)
        print(f"AI response: {response_text}")

        print("Converting AI response to speech...")
        _, audio_thread = TTS_Mozilla(response_text, output_audio_path)
        print("AI is speaking...")
        audio_thread.join()
        print("AI finished speaking.")

if __name__ == "__main__":
    interactive_session()