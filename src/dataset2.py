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
        
        while(True):
            try:
                sample = self.hf_dataset[idx]
                pattern = re.compile(r'[a-zA-Z+=-]')

                if(len(sample['speech'])/16000 > 10.0 or len(sample['answer']) > 1500):
                    # print("skip data")
                    idx = idx+1
                    continue
                # elif(pattern.search(sample['trascript'])):
                #         # 搜索字符串中是否包含这些字符
                #     idx = idx+1
                #     continue
                break
            except:
                idx = idx+1
        
        if('speech' in sample.keys()):
            answer = sample['answer']
            audio = sample['speech']
        
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
        