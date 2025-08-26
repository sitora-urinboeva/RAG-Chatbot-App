import numpy as np
from transformers import AutoProcessor, BarkModel
import os
import scipy
import nltk
from nltk.tokenize import sent_tokenize
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models


nltk.download('punkt')
model_folder = './suno_bark_model'
voice_preset = 'v2/ru_speaker_5'

class TTS_Generator():
    def __init__(self, voice='v2/ru_speaker_5', model_path='./suno_bark_model'):
        self.model_path = model_path
        self.voice = voice
        self.sample_rate = SAMPLE_RATE
        self.processor, self.model = self.load_model_processor()

    def load_model_processor(self, model_path='./suno_bark_model'):
        if os.path.exists(model_path):
            print('Loading the model from local directory...')
            processor = AutoProcessor.from_pretrained(model_path)
            model = BarkModel.from_pretrained(model_path)
        else:
            print('Downloading the model from huggingface...')
            processor = AutoProcessor.from_pretrained("suno/bark-small")
            model = BarkModel.from_pretrained("suno/bark-small")
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)

        return processor, model

    def generate_audio(self, text, voice):
        inputs = self.processor(text, voice_preset=voice, return_tensors='pt')
        audio = self.model.generate(**inputs)
        audio = audio.cpu().numpy().squeeze()
        return audio

    def audio_synthesis(self, text, voice, output_file='output.wav'):

        if len(text) >= 100:
            print('Text is too long! Splitting into sentences...')
            sentences = sent_tokenize(text, language='russian')
            silence_duration = 0.25
            silence_duration = np.zeros(int(silence_duration * SAMPLE_RATE))
            parts = []
            for part in sentences:
                print('Processing:', part)
                audio = self.generate_audio(part, voice)
                parts.extend([audio, silence_duration])
            full_audio = np.concatenate(parts)
        else:
            print('Generating audio...')
            full_audio = self.generate_audio(text, voice)

        scipy.io.wavfile.write(output_file, data=full_audio.astype(np.float32()), rate=SAMPLE_RATE)
        print('Audio file is saved!')
