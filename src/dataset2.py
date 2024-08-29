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
                break
            except:
                idx = idx+1
        
        if(self.aishell_transcipt):
            path = 'temp_datasets/aishell/data_aishell/wav/train/'
            sr, audio = wav.read(path+sample+".wav")
            audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=16000)
            answer = self.aishell_transcipt[sample]
            answer = answer.replace(" ","")
            
        elif('translation'in sample.keys()):
            #covost2
            
            sentence = sample['sentence']
            if(sentence[0] == '\"' and sentence[len(sentence)-1] == '\"'):
                sentence = sentence[1:-1]
            sentence = sentence.strip()
            # answer = sentence
            answer = sentence +'$'+ sample['translation']
            # answer = sample['translation']
            audio = sample['audio']['array']
            # audio = resampy.resample(audio, 48000, 16000)
            # audio = librosa.resample(audio,orig_sr= 48000,target_sr= 16000)
            
        elif('sentence' in sample.keys()):
            #common voice
            
            answer = sample['sentence']
            audio = sample['audio']['array']
            audio = resampy.resample(audio, 48000, 16000)
        elif('transcipt' in sample.keys()):
            #multilingual-librispeech
            audio = sample['audio']['array']
            answer = sample['transcript']
        elif('audio' in sample.keys()):
            #librispeech
            audio = sample['audio']['array']
            answer = sample['transcript']
        else:
            #en-final
            audio = sample['speech']
            answer = sample['text']
        
        # print(f"speech input{idx}:{len(audio)}")

        return audio, answer.lower()
        