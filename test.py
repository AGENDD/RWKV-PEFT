from datasets import load_dataset,load_from_disk, concatenate_datasets,Features, Sequence, Value
import torchaudio
import torch
# from melo.api import TTS
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
import re
import torch
import glob
import augment, random

ds = load_from_disk('temp_datasets/chinese_speech_only_cosy')


def audioAug(audio):
    
    random_speed = random. uniform(0.7, 1.3)
    audio = np.array(audio)
    audio = librosa.effects.time_stretch(audio, random_speed)
    audio = audio.tolist()
    
    x = torch.tensor(audio)
    x = x.unsqueeze(0)
    
    


    sr = 16000
    
    random_pitch_shift = lambda: np.random.randint(-400, +400)
    random_room_size = lambda: np.random.randint(0, 101)
    # random_noise = lambda: torch.zeros_like(x).uniform_()
    random_noise = lambda: torch.zeros_like(x).uniform_() * np.random.uniform(0, 0.3)
    random_dropout = random.uniform(0, 0.2)
    
    
    combination = augment.EffectChain() \
        .pitch("-q", random_pitch_shift).rate(sr) \
        .time_dropout(max_seconds=random_dropout) \
        .reverb(50, 50, random_room_size).channels(1) \
        .additive_noise(random_noise, snr=15) 
        
    
    y = combination.apply(x, src_info={'rate': sr}, target_info={'rate': sr})
    
    y = list(y[0])
    

    
    return y


x = ds[0]['speech_cosy'][0]

for i in tqdm(range(100)):
    audio = audioAug(x)
    sf.write(f"temp_audios/audio{i}.wav", audio, 16000)



# print(ds)
# def mapp(example):
#     transcript = example['trascript']
    
#     pattern = re.compile(r'[a-zA-Z+=-]')
#     # 搜索字符串中是否包含这些字符
#     if pattern.search(transcript):
#         example['transcript'] = None
#     else:
#         example['transcript'] = transcript
    
#     return example

# ds = ds.map(mapp,num_proc=32,remove_columns=['trascript'],cache_file_name="cache/file.arrow")

# def fill(example):
#     if(example['transcript'] == None):
#         return False
    
#     return True

# ds = ds.filter(fill, num_proc=32)

# print(ds)
# ds.save_to_disk('temp_datasets/chinese_speech_only')

exit(0)
# # 查找以“rwkv”开头的文件
# file_list = glob.glob('output/rwkv*.pth')

# # 确保找到文件
# if file_list:
#     file_path = file_list[0]  # 假设只有一个匹配的文件
#     print(f"Loading file: {file_path}")

#     # 加载模型参数
#     model_state_dict = torch.load(file_path)

#     # 打印所有参数的名字
#     for name, param in model_state_dict.items():
#         print(name)
# else:
#     print("No file starting with 'rwkv' found.")


# ds = load_from_disk("temp_datasets/ZHEN_mixed_filtered").shuffle()

# # i = 0
# # for data in ds:
# #     audio = data['speech']
# #     i+=1
# #     sf.write(f'temp_audios/output{i}.wav', audio, 16000)
    

# def mapp(example):
#     if(len(example['speech']) / 16000 > 15.0 or len(example['answer']) > 1000):
#         example['speech'] = None


# ds = ds.map(mapp, num_proc=32)

# def fill(example):
#     if(example['speech'] == None):
#         return False
    
#     return True

# ds = ds.filter(fill, num_proc=32)
# print(ds)
# ds.save_to_disk("temp_datasets/ZHEN_mixed_filteredd")

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

# from contextlib import contextmanager, redirect_stdout, redirect_stderr
# from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
# from cosyvoice.utils.file_utils import load_wav
# import torchaudio
# import random
# from datasets import load_from_disk
# import librosa
# import os
# import numpy as np

# @contextmanager
# def suppress_stdout(*args, **kwargs):
#     with open(os.devnull, 'w') as devnull:
#         with redirect_stdout(devnull), redirect_stderr(devnull):
#             yield


# ds = load_from_disk("~/JRwork/RWKV-PEFT/temp_datasets/chinese_speech_only").select(range(50000,60000))
# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)

# print(ds)
# def mapp(sample):
#     random_number = random.randint(0, 99)
#     prompt_speech_16k = load_wav(f'temp_audios/audio{random_number}.wav', 16000)

#     try:
#         with suppress_stdout():
#             for i, j in enumerate(cosyvoice.inference_instruct2(sample['transcript'], '', prompt_speech_16k, stream=False)):
#                 # torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
#                 cosy = librosa.resample(np.array(j['tts_speech']), orig_sr=cosyvoice.sample_rate, target_sr=16000)
#         sample['speech_cosy'] = cosy
#     except:
#         sample['speech_cosy'] = None

#     return sample

# ds = ds.map(mapp,cache_file_name="cache/file.arrow")

# def fill(sample):
#     if(sample['speech_cosy'] == None):
#         return False
#     return True

# ds = ds.filter(fill, num_proc=32)

# print(ds)

# ds.save_to_disk("~/JRwork/RWKV-PEFT/temp_datasets/chinese_speech_only_cosy6")

