from transformers import AutoModelForCausalLM, AutoTokenizer

language_tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-6-world-1b6",trust_remote_code=True)

# token = language_tokenizer(
#                 "<s>",
#                 return_tensors="pt",
#             )


token = language_tokenizer.decode([-100])
token2 = language_tokenizer.decode([0])

print(token)
print(token2)
