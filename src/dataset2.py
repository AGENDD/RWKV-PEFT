########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime
from rwkv.utils import PIPELINE
import librosa
import resampy
import scipy.io.wavfile as wav
import re

pipeline = PIPELINE('rwkv6', "rwkv_vocab_v20230424")

class MyDataset(Dataset):
    def __init__(self, args, hf_dataset, aishell_transcipt = None):
        self.args = args
        self.hf_dataset = hf_dataset
        self.aishell_transcipt = aishell_transcipt
    
        
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        
        def audioAug(audio):

            audio = np.array(audio)
            sr = 16000
            ######################时域拉伸
            random_speed = random.uniform(0.7, 1.3)
            
            audio = librosa.effects.time_stretch(audio, rate = random_speed)
            # audio = audio.tolist()
                
            ######################音高变化
            n_steps = np.random.uniform(-4, 4)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
            ######################时域遮挡
            mask_duration = np.random.uniform(0, 0.2)
            mask_length = int(mask_duration * sr)
            mask_start = np.random.randint(0, len(audio) - mask_length)
            audio[mask_start:mask_start + mask_length] = 0
            
            ######################加噪
            noise_level = random_speed = random.uniform(0.0001, 0.001)
            noise = np.random.randn(len(audio))
            audio = audio + noise_level * noise
            
            audio = audio.tolist()
            return audio
        # sample = self.hf_dataset[idx]
        while(True):
            try:
                QA = True
                sample = self.hf_dataset[idx]
                if(QA):
                    if(len(sample['audio']['array'])/16000 > 15.0):
                        # print("skip data audio too long")
                        idx = idx+1
                        continue
                    if(len(sample['answer']) > 1500):
                        # print("skip data answer too long")
                        idx = idx+1
                        continue
                
                # if(len(sample['audio']['array'])/16000 > 15.0):
                #     # print("skip data audio too long")
                #     idx = idx+1
                #     continue
                                
                # pattern = re.compile(r'[a-zA-Z+=-]')
                # if(pattern.search(sample['trascript'])):
                #         # 搜索字符串中是否包含这些字符
                #     idx = idx+1
                #     continue
                break
            except:
                idx = idx+1
        if('messages' in sample.keys()):
            audio = sample['audio']['array']
            answer = sample['messages'][1]['content']      
        elif('transcription' in sample.keys()):
            #aishell
            audio = sample['audio']['array']
            answer = sample['transcription']
            answer = answer.replace(" ","")
        elif('transcript' in sample.keys()):
            audio = sample['audio']['array']
            answer = sample['transcript']+" ~ "+sample['answer']
        elif('speech' in sample.keys()):
            answer = sample['transcript']+" ~ "+sample['answer']
            audio = sample['speech_cosy'][0]
            
            # try:
            #     audio = audioAug(audio)
            # except:
            #     audio = audio
        # elif('split_name' in sample.keys()):
        #     #Voice assistant
        #     answer = sample['answer']
        #     # words = answer.split()
        #     # answer = words[:64]
        #     # answer = " ".join(answer)
            
        #     audio = sample['question_audio']['array']
        #     audio = resampy.resample(audio, 22050, 16000)
            
        # if(self.aishell_transcipt):
        #     #aishell
        #     path = 'temp_datasets/aishell/data_aishell/wav/train/'
        #     sr, audio = wav.read(path+sample+".wav")
        #     audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=16000)
        #     answer = self.aishell_transcipt[sample]
        #     answer = answer.replace(" ","")

        return audio, answer
        