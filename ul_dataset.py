from datasets import load_from_disk,DatasetDict, concatenate_datasets, Audio

dataset = load_from_disk("temp_datasets/chinese_speech_only_cosy")
dataset2 = load_from_disk("temp_datasets/chinese_speech_only_cosy2")
dataset3 = load_from_disk("temp_datasets/chinese_speech_only_cosy3")
dataset4 = load_from_disk("temp_datasets/chinese_speech_only_cosy4")
dataset5 = load_from_disk("temp_datasets/chinese_speech_only_cosy5")
dataset6 = load_from_disk("temp_datasets/chinese_speech_only_cosy6")
dataset7 = load_from_disk("temp_datasets/chinese_speech_only_cosy7")
dataset = concatenate_datasets([dataset, dataset2,dataset3,dataset4,dataset5,dataset6,dataset7]).shuffle()#49999


token = "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" #write


from huggingface_hub import login

# 登录 Hugging Face
login(token)

def mapp(data):
    data['audio'] = {'array': data['speech_cosy'][0], 'sample_rate':16000}
    return data
    
dataset = dataset.remove_columns(['speech'])
print("mapping")
dataset = dataset.map(mapp, num_proc = 16, cache_file_name="cache/file.arrow")
dataset = dataset.remove_columns(['speech_cosy'])
print("Custing...")
dataset = dataset.cast_column('audio', Audio())


# 创建 DatasetDict 对象
dataset_dict = DatasetDict({'train': dataset})

# 上传数据集
while(True):
    try:
        dataset_dict.push_to_hub('chinese_speech_cosy_audio')
        break
    except Exception as e:
        print(e)
        
print("finish")