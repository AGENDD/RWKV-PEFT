from datasets import load_dataset, load_from_disk
token = "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"

ds = load_from_disk("temp_datasets/ultrachat")

print(ds)
print(ds[0])


print("finish")