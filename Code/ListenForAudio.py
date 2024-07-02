import pyaudio
import wave
import time
import threading

def record_audio(output_path, max_duration=30, chunk=1024, sample_format=pyaudio.paInt16, channels=2, fs=44100):
    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print(f"Recording... Press Enter to stop (max duration: {max_duration} seconds).")

    frames = []
    recording = True
    start_time = time.time()

    def input_thread():
        nonlocal recording
        input()
        recording = False

    # Start input thread
    threading.Thread(target=input_thread, daemon=True).start()

    while recording:
        if time.time() - start_time > max_duration:
            print("Maximum recording duration reached.")
            break
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording stopped.")

    wf = wave.open(output_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to {output_path}")

    return output_path

# if __name__ == "__main__":
#     output_path = "Neuphonic/Data/Recordings/recorded_audio.wav"
#     record_audio(output_path)