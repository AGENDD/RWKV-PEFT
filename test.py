from datasets import load_dataset,load_from_disk
import torchaudio
import torch
from melo.api import TTS
import io
import soundfile as sf
import resampy
import numpy as np
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os
import librosa
from torchaudio.transforms import Resample
from torchaudio import load, save

@contextmanager
def suppress_stdout(*args, **kwargs):
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


dataset = load_from_disk("temp_datasets/ultrachat")

TTS = TTS(language='EN', device='auto')
speaker_ids = TTS.hps.data.spk2id


def fun(example):
    try:
        with suppress_stdout():
            transcript = example['prompt']

            wave = TTS.tts_to_file(transcript, speaker_ids['EN-US'], None, speed=1.0)
            wave = torch.tensor(wave).unsqueeze(0)
            resample = Resample(44100, 16000)
            resampled_audio = resample(wave[0])
            wave = resampled_audio.squeeze(0).numpy()
        example['speech'] = wave
    except:
        example['speech'] = None
    return example

dataset = dataset.map(fun,remove_columns=["prompt_id"])

def fil(example):
    if(example['speech'] == None):
        print(example['prompt'])
        return False
    else:
        return True
    
dataset = dataset.filter(fil,num_proc=32)


print(len(dataset))

dataset.save_to_disk("temp_datasets/ultrachat_speech")


# dataset = load_from_disk("temp_datasets/ultrachat").select(range(10))



# for i,data in enumerate(dataset):
#     transcript = data['prompt']

#     wave = TTS.tts_to_file(transcript, speaker_ids['EN-US'], None, speed=1.0)
#     sf.write("temp_audios/origin.wav",wave, 44100)
#     print(f"{len(wave)}:{wave}")
#     wave = torch.tensor(wave).unsqueeze(0)
#     resample = Resample(44100, 16000)
#     resampled_audio = resample(wave[0])
#     wave = resampled_audio.squeeze(0).numpy()
#     sf.write("temp_audios/resempled.wav",wave, 16000)
#     print(f"{len(wave)}:{wave}")
#     with io.BytesIO() as buffer:
#         sf.write(buffer, wave, 16000, format='WAV')
#         buffer.seek(0)
#         wave, sr = sf.read(buffer,dtype='float32')
#     print(f"{len(wave)}:{wave}")
#     sf.write("temp_audios/normal.wav",wave, sr)

    
#     break
            