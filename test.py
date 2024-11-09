from datasets import load_dataset,load_from_disk, concatenate_datasets,Features, Sequence, Value
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
from tqdm import tqdm

ds1 = load_from_disk("temp_datasets/chinese_speech").select(range(10))
# 修正 ds1 的拼写错误
ds1 = ds1.rename_column('trascript', 'transcript')

# ds2 = load_from_disk("temp_datasets/VoiceAssistant").select(range(123433))
ds2 = load_from_disk("temp_datasets/VoiceAssistant").select(range(10))

print(type(ds1[0]['speech']))
print(type(ds2[0]['question_audio']['array']))

print(type(ds1[0]['speech'][0]))
print(type(ds2[0]['question_audio']['array'][0]))


def mapp(sample):

    audio = sample['question_audio']['array'].astype(np.float32)
    sample['speech'] = resampy.resample(audio, 22050, 16000).tolist()
    sample['transcript'] = sample['question']
    
    
    return sample

arr = ds2.column_names

arr.remove('answer')

ds2 = ds2.map(mapp,remove_columns=arr)



# 定义一致的特征
features = Features({
    'speech': Sequence(feature=Value('float32')),
    'transcript': Value('string'),
    'answer': Value('string')
})



print(type(ds1[0]['speech']))
print(type(ds2[0]['speech']))

print(type(ds1[0]['speech'][0]))
print(type(ds2[0]['speech'][0]))

print(ds1)
print(ds2)

# 获取 ds1 的列顺序
ds1_columns = ds1.column_names

# 重新排列 ds2 的列顺序
ds2 = ds2.map(lambda example: {col: example[col] for col in ds1_columns})

# 将数据集转换为一致的特征
ds1 = ds1.cast(features)
ds2 = ds2.cast(features)

ds = concatenate_datasets([ds1,ds2])

# ds.save_to_disk("temp_datasets/ZHEN_mixed")
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


# # dataset = load_from_disk("temp_datasets/ultrachat")
# dataset = load_dataset("Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese")['train']
# TTS = TTS(language='ZH', device='cuda')
# speaker_ids = TTS.hps.data.spk2id


# def fun(example):
#     try:
#         with suppress_stdout():
#             if(int(example['instruction_length']) > 50):
#                 # print("Too long:",end="")
#                 raise ValueError

#             transcript = example['instruction']

#             wave = TTS.tts_to_file(transcript, speaker_ids['ZH'], None, speed=1.0)
#             wave = torch.tensor(wave).unsqueeze(0)
#             resample = Resample(44100, 16000)
#             resampled_audio = resample(wave[0])
#             wave = resampled_audio.squeeze(0).numpy()
#         example['speech'] = wave
#     except:
#         # print(example['instruction'])
#         example['speech'] = None
#     example['trascript'] = example['instruction']
#     example['answer'] = example['response']
#     return example
# print("start")
# dataset = dataset.map(fun,remove_columns=dataset.column_names)
# try:
#     dataset.save_to_disk("temp_datasets/chinese_speechQA_unfiltered")
# except:
#     print("cannot save unfiltered")
    
# def fil(example):
#     if(example['speech'] == None):
#         return False
#     else:
#         return True
    
# dataset = dataset.filter(fil)


# print(len(dataset))

# dataset.save_to_disk("temp_datasets/chinese_speech")


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
            