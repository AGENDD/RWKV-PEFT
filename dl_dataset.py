from datasets import load_dataset

arr = ['dutch','french','german','italian','polish','portuguese','spanish']

for i in arr:
    print(i)
    while(True):
        try:
            mls = load_dataset("facebook/multilingual_librispeech", i, split="dev",resume_download=True)
            break
        except:
            continue

print("finish")