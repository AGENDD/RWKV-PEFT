from datasets import load_dataset, load_from_disk
from torchaudio.transforms import Resample
from torchaudio import load, save
import torch
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os

from tqdm import tqdm
import numpy as np
import soundfile as sf

# token = "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"

ds = load_from_disk("temp_datasets/chinese_speech_only_cosy").shuffle()

for i in tqdm(range(100)):
    array = ds[i]['speech_cosy']
    sf.write(f'temp_audios/audio{i}.wav', array, 22050)

