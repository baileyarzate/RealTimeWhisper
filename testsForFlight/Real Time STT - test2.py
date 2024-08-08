from RealtimeSTT import AudioToTextRecorder
import os

if __name__ == '__main__':

    print("Initializing FlyWhisper...")


    full_sentences = []
    displayed_text = ""

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    def text_detected(text):
        global displayed_text
        sentences_with_style = [
            f"{sentence} "
            for i, sentence in enumerate(full_sentences)
        ]
        new_text = "".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

        if new_text != displayed_text:
            displayed_text = new_text
            clear_console()
            print(displayed_text, end="", flush=True)

    def process_text(text):
        full_sentences.append(text)
        text_detected("")

    recorder_config = {
        'spinner': False,
        'model': 'distil-large-v2',
        'language': 'sp',
        'silero_sensitivity': 0.4,
        'webrtc_sensitivity': 2,
        'post_speech_silence_duration': 0.4,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.2,
        'realtime_model_type': 'tiny.en',
        'on_realtime_transcription_update': text_detected, 
        'device': "cuda",
    }

    recorder = AudioToTextRecorder(**recorder_config)

    clear_console()
    print("Transcription has begun...", end="", flush=True)

    while True:
        recorder.text(process_text)