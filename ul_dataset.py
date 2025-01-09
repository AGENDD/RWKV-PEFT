from datasets import load_from_disk,DatasetDict, concatenate_datasets

dataset = load_from_disk("temp_datasets/chinese_speech_only_cosy")
dataset2 = load_from_disk("temp_datasets/chinese_speech_only_cosy2")
dataset3 = load_from_disk("temp_datasets/chinese_speech_only_cosy3")
dataset4 = load_from_disk("temp_datasets/chinese_speech_only_cosy4")
dataset5 = load_from_disk("temp_datasets/chinese_speech_only_cosy5")
dataset6 = load_from_disk("temp_datasets/chinese_speech_only_cosy6")
dataset7 = load_from_disk("temp_datasets/chinese_speech_only_cosy7")
dataset = concatenate_datasets([dataset, dataset2,dataset3,dataset4,dataset5,dataset6,dataset7]).shuffle()#49999

dataset = dataset.remove_columns(['speech'])
print(dataset)

token = "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" #write


from huggingface_hub import login

# 登录 Hugging Face
login(token)

# 创建 DatasetDict 对象
dataset_dict = DatasetDict({'train': dataset})

# 上传数据集
while(True):
    try:
        dataset_dict.push_to_hub('chinese_speech_cosy')
        break
    except Exception as e:
        print(e)
        
print("finish")