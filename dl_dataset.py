from datasets import load_dataset
token = "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"


while(True):
    ds = load_dataset("gpt-omni/VoiceAssistant-400K", token = token, num_proc=32)
    break
print("finish")