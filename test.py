import librosa
from datasets import load_from_disk,load_dataset, concatenate_datasets
import soundfile as sf
dataset = load_dataset('covost2','en_zh-CN',data_dir = 'temp_datasets/covost-en_zhCN')["train"].select(range(100))



with open("temp_audios/text.txt",'w') as f:
    
    for i in range(100):    
        sample = dataset[i]
        audio = sample['audio']['array']
        audio = librosa.resample(audio,orig_sr= 48000,target_sr= 16000)
        
        transcription = sample['sentence']
        translation = sample["translation"]
        
        f.write(f"{i}")
        f.write(f"{transcription}\n")
        f.write(f"{translation}\n")
        sf.write(f'{i}.wav', audio, 16000)
        
        