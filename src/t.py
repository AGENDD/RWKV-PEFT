from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 假设 tensors[i] 和 transcriptions_with_eoa_embed[i] 分别是 [894, 2560] 和 [322, 2560] 的张量
tensors_i = torch.randn(894, 2560).to("cuda", torch.bfloat16)
transcriptions_with_eoa_embed_i = torch.randn(322, 2560).to("cuda", torch.bfloat16)

# 确保所有张量都在同一个设备上
tensors_i = tensors_i.to("cuda")
transcriptions_with_eoa_embed_i = transcriptions_with_eoa_embed_i.to("cuda")

# 检查数据类型和形状
print(tensors_i.dtype, transcriptions_with_eoa_embed_i.dtype)
print(tensors_i.shape, transcriptions_with_eoa_embed_i.shape)

# 执行拼接操作
result = torch.cat([tensors_i, transcriptions_with_eoa_embed_i], dim=0)

# 检查拼接后的形状
print(result.shape)  # 应该输出 torch.Size([1216, 2560])
