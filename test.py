import librosa
from datasets import load_from_disk,load_dataset, concatenate_datasets
import soundfile as sf
from tqdm import tqdm
dataset = load_dataset('covost2','en_zh-CN',data_dir = 'temp_datasets/covost-en_zhCN')["train"].select(range(10))

import resampy

with open("temp_audios/text.txt",'w') as f:
    
    for i in tqdm(range(10)):    
        sample = dataset[i]
        audio = sample['audio']['array']
        tqdm.write(f"before resample{len(audio)}")
        # audio = librosa.resample(audio,orig_sr= 48000,target_sr= 16000)
        audio = resampy.resample(audio, 48000, 16000)
        tqdm.write(f"after resample{len(audio)}")
        transcription = sample['sentence']
        translation = sample["translation"]
        
        f.write(f"{i}\n")
        f.write(f"{transcription}\n")
        f.write(f"{translation}\n")
        sf.write(f'temp_audios/{i}.wav', audio, 16000)
        
        