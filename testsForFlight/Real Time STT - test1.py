from RealtimeSTT import AudioToTextRecorder
import torch
from multiprocessing import freeze_support
global recorder
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
def main():
    def process_text(text):
        print (text)
        
    with AudioToTextRecorder(model = 'tiny',
                                   language = 'en',
                                   compute_type = 'int8', #quantization: https://opennmt.net/CTranslate2/quantization.html
                                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
                                   beam_size = 5, #smaller: faster, less accurate;  larger is the inverse
                                   enable_realtime_transcription = True,
                                   realtime_model_type = 'tiny',
                                   realtime_processing_pause = 0.2,
                                   beam_size_realtime = 3,
                                   post_speech_silence_duration=1.5,
                                   spinner = False) as recorder:
        print("Transcription has begun:")
        print("------------------------")
        try:
            while True:
                recorder.text(process_text)
        except KeyboardInterrupt:
            recorder.stop()
            recorder.shutdown()
            print("Interrupted by user. Stopping transcription...")
            
if __name__=='__main__':
    freeze_support()
    main()