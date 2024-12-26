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

ds = load_from_disk("temp_datasets/VoiceAssistant")

for i in tqdm(range(100)):
    array = ds[i]["question_audio"]['array']
    sf.write('temp_audios/audio{i}.wav', array, 22050)

