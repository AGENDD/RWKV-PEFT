from datasets import load_from_disk,DatasetDict

dataset = load_from_disk("temp_datasets/ultrachat_tensor_10000")

token = "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" #write


from huggingface_hub import login

# 登录 Hugging Face
login(token)

# 创建 DatasetDict 对象
dataset_dict = DatasetDict({'train': dataset})

# 上传数据集
dataset_dict.push_to_hub('ultrachat_tensor_10k')
print("finish")