from datasets import load_dataset, load_from_disk
from melo.api import TTS
from torchaudio.transforms import Resample
from torchaudio import load, save
import torch
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import os


# token = "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"

# ds = load_from_disk("temp_datasets/ultrachat")

# TTS = TTS(language='EN', device='auto')
# speaker_ids = TTS.hps.data.spk2id


# @contextmanager
# def suppress_stdout(*args, **kwargs):
#     with open(os.devnull, 'w') as devnull:
#         with redirect_stdout(devnull), redirect_stderr(devnull):
#             yield

# def mapp(data):
#     messages = data['messages']
#     count = 0
    
#     speech_messages = []
#     respond_messages = []
#     for message in messages:
#         if(message["role"] == "assistant"):
#             assistantdic = {}
#             assistantdic['role'] = 'assistent'
#             assistantdic['content'] = message['content']
#             respond_messages.append(assistantdic)
#         else:
#             leng = len(message["content"].split())
#             if(leng > 30):
#                 break
#             else:
#                 try:
#                     with suppress_stdout():
#                         wave = TTS.tts_to_file(message["content"], speaker_ids['EN-US'], None, speed=1.0)
#                         wave = torch.tensor(wave).unsqueeze(0)
#                         resample = Resample(44100, 16000)
#                         resampled_audio = resample(wave[0])
#                         wave = resampled_audio.squeeze(0).numpy()
                        
#                     if(len(wave) / 16000 > 15.0):
#                         break
                    
#                     userdic = {'content': wave, 'role': "user", 'transcript': message['content']}
#                     speech_messages.append(userdic)
#                     count += 1
#                 except Exception as e:
#                     print(f"Error processing message: {e}")
#                     print(f"data: {message}")
#                     count = -1
#                     break
    
#     data["turns"] = count
#     data['speech_messages'] = speech_messages
#     data['respond_messages'] = respond_messages
#     return data

# ds = ds.map(mapp, remove_columns=['messages', 'prompt', 'prompt_id'])

# ds.save_to_disk("temp_datasets/ultrachat_speech_multiTurns_unfiltered")


ds = load_from_disk("temp_datasets/ultrachat_speech_multiTurns_unfiltered")


def fil(data):
    if(data["turns"] == -1 or data["turns"] == 0):
        return False
    
    return True

ds = ds.filter(fil)
ds.save_to_disk("temp_datasets/ultrachat_speech_multiTurns")



print("finish")