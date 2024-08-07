import librosa
from datasets import load_from_disk,load_dataset, concatenate_datasets
import soundfile as sf
from tqdm import tqdm
dataset = load_dataset('covost2','en_zh-CN',data_dir = 'temp_datasets/covost-en_zhCN')["train"].select(range(10))
from scipy.io.wavfile import write

import resampy

with open("temp_audios/text.txt",'w') as f:
    
    for i in tqdm(range(10)):    
        sample = dataset[i]
        audio = sample['audio']['array']
        tqdm.write(f"before resample:{len(audio)}")
        audio1 = librosa.resample(audio,orig_sr= 48000,target_sr= 16000)
        audio2 = resampy.resample(audio, 48000, 16000)
        tqdm.write(f"after resample1:{len(audio1)}")
        tqdm.write(f"after resample2:{len(audio2)}")
        transcription = sample['sentence']
        translation = sample["translation"]
        
        f.write(f"{i}\n")
        f.write(f"{transcription}\n")
        f.write(f"{translation}\n")
        sf.write(f'temp_audio/temp{i}_1.wav', audio1, 16000)
        sf.write(f'temp_audio/temp{i}_2.wav', audio2, 16000)
        sf.write(f'temp_audio/temp{i}_3.wav', audio, 48000)
        
        