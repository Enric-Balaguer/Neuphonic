import whisper
import pyaudio
import numpy as np

def ASR_audio_file(audio_file_path:str):
    """
    Converts audio to text
    audio_file_path: string containing the path of the audio to be converted to text.
    """
    model = whisper.load_model("base")

    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return result.text

def ASR_real_time():
    # Load model
    model = whisper.load_model("base")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Define stream parameters
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK_SIZE = 1024
    BUFFER_DURATION = 5  # Duration in seconds to buffer audio
    BUFFER_SIZE = int(RATE / CHUNK_SIZE * BUFFER_DURATION)

    # Setup audio input stream
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

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
                result = model.transcribe(audio_buffer, temperature=0)
                print(f"Transcribed Text: {result['text']}")
                # Clear the buffer or handle as needed
                audio_buffer = np.empty(0, dtype=np.float32)

    except KeyboardInterrupt:
        print("Stopping...")

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()

    return result