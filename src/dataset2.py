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
pipeline = PIPELINE('rwkv6', "rwkv_vocab_v20230424")

class MyDataset(Dataset):
    def __init__(self, args, hf_dataset):
        self.args = args
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        if('sentence' in sample.keys()):
            #common voice
            answer = sample['sentence']
            audio = sample['audio']['array']
            audio = librosa.resample(audio,original_sr= 48000,target_sr= 16000)
        else:
            #librispeech
            audio = sample['speech']
            answer = sample['text']
        
        # print(f"speech input{idx}:{len(audio)}")
        return audio, answer.lower()
        