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

count = {}
ds = load_from_disk("temp_datasets/ultrachat_speech_multiTurns")

for data in ds:
    if(data['turns'] not in count.keys()):
        count[data['turns']] = 1
    else:
        count[data['turns']] += 1

print(count)
#########################################################################################################

# # 加载数据集
# ds = load_dataset("gpt-omni/VoiceAssistant-400K")['train']  # 假设这里是'train' split

# def fun(x):
    
#     if(x['split_name'] == 'identity'):
#         return False
#     elif(len(x['question_audio']["array"]) / 22050 >= 15.0):
#         return False
        
#     return True


# # 过滤掉 split_name 为 identity 的数据，并启用多进程
# filtered_ds = ds.filter(fun, num_proc=32)

# # 去掉 index, round, answer_snac 这三列数据
# filtered_ds = filtered_ds.remove_columns(['index', 'round', 'answer_snac'])

# # 保存新的数据集
# filtered_ds.save_to_disk("temp_datasets/VoiceAssistant")

###############################################################################################################
# @contextmanager
# def suppress_stdout(*args, **kwargs):
#     with open(os.devnull, 'w') as devnull:
#         with redirect_stdout(devnull), redirect_stderr(devnull):
#             yield


# dataset = load_from_disk("temp_datasets/ultrachat")

# TTS = TTS(language='EN', device='auto')
# speaker_ids = TTS.hps.data.spk2id


# def fun(example):
#     try:
#         with suppress_stdout():
#             transcript = example['prompt']

#             wave = TTS.tts_to_file(transcript, speaker_ids['EN-US'], None, speed=1.0)
#             wave = torch.tensor(wave).unsqueeze(0)
#             resample = Resample(44100, 16000)
#             resampled_audio = resample(wave[0])
#             wave = resampled_audio.squeeze(0).numpy()
#         example['speech'] = wave
#     except:
#         example['speech'] = None
#     return example

# dataset = dataset.map(fun,remove_columns=["prompt_id"])

# def fil(example):
#     if(example['speech'] == None):
#         print(example['prompt'])
#         return False
#     else:
#         return True
    
# dataset = dataset.filter(fil,num_proc=32)


# print(len(dataset))

# dataset.save_to_disk("temp_datasets/ultrachat_speech")


#############################################################################################################

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
            