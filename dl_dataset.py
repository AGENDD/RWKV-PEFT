from datasets import load_dataset, load_from_disk
from torchaudio.transforms import Resample
from torchaudio import load, save
import torch
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os


# token = "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"
while(True):
    try:
        ds = load_dataset("carlot/AIShell", cache_dir='temp_datasets')
        break
    except Exception as e:
        print(e)

print(ds)