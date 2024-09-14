from datasets import load_dataset


dataset = load_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft",cache_dir="temp_cache")#207865
        
# 定义过滤函数
def filter_long_prompts(example):
    return len(example['prompt'].split()) <= 30

# 过滤数据集，使用32个线程
filtered_dataset = dataset.filter(filter_long_prompts, num_proc=32)

print(len(filtered_dataset))

filtered_dataset.save_to_disk("temp_datasets/ultrachat")

