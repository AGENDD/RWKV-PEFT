import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoProcessor, AutoModel


class SpeechAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpeechAdapter, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=3072, kernel_size=3, stride=2)
        self.transformer = nn.TransformerEncoderLayer(d_model=3072, nhead=8, dim_feedforward=4096)
        self.linear = nn.Linear(3072, output_dim)
    def forward(self, x, mask):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        # x shape: (batch_size, input_dim, seq_len)
        x = self.conv(x)
        # x shape after conv: (batch_size, input_dim, new_seq_len)
        x = x.permute(2, 0, 1)  # Transformer expects (seq_len, batch_size, input_dim)
        mask = mask[:, : x.shape[0]]
        # x = self.transformer(x, src_key_padding_mask=mask.bool())
        x = self.transformer(x, src_key_padding_mask=~mask.bool())
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, input_dim)
        x = self.linear(x)
        return x, mask



class SpeechEncoder(nn.Module):
    def __init__(
        self,
        model_id,
        project_dim,
        downsample_K=5,
        hidden_dim=2048,
        train_mode="adapter",
        device="cuda",
    ):
        assert train_mode in ["adapter", "full"]
        super(SpeechEncoder, self).__init__()

        self.device = device
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except:
            self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.time_reduction_factor = int(
            self.processor.feature_extractor.sampling_rate / 50
        )
        self.padding_length = 320
        
        # config_path = "temp_models/ST/config.json"
        # ckpt_path = "temp_models/ST/SpeechTokenizer.pt"
        
        
        # self.model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).eval()
        # self.model = self.model.to(self.device,dtype=torch.bfloat16)
        self.model = AutoModel.from_pretrained(model_id).to(self.device,dtype=torch.bfloat16)
        # self.model = AutoModel.from_pretrained(model_id,cache_dir="temp_models").to(self.device,dtype=torch.bfloat16)
        self.model.eval()
        self.model_output_dim = self.model.config.hidden_size
        self.downsample_K = downsample_K
        self.project_dim = project_dim
        if hidden_dim is None:
            self.hidden_dim = self.project_dim * 2
        else:
            self.hidden_dim = hidden_dim
            
            
        # self.downsample_K = downsample_K
        
        # self.model_output_dim = self.model.config.hidden_size
        # self.model_output_dim = self.model.n_q
        # self.downsample_K = downsample_K
        self.project_dim = project_dim

            
        self.adapter = SpeechAdapter(self.model_output_dim, self.project_dim).to(self.device,dtype=torch.bfloat16)
        self.set_gradient(train_mode)
        
        # print("Parameters in speech encoder that require grad:")

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"    {name}: {param.shape}")
        

    def set_gradient(self, train_mode):
        """
        if train_mode is "adapter", only train the adapter layers, otherwise train the whole model
        """
        if train_mode == "adapter":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.adapter.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.adapter.parameters():
                param.requires_grad = True

    def downsample_mask(self, mask):
        """
        Also need to handle the masking issue, to let the model not to attend to the padding tokens
        """
        attention_mask = mask  # [batch, num_samples]
        # create the mask
        mask = attention_mask[:, :: (self.downsample_K)]
        # mask = attention_mask[:, :: (self.time_reduction_factor * )]
        return mask


    def padding_mask(self, x):
        #x:List[float]
        #return x:tensor(B,channel,seq_length) , mask:tensor(B,seq_length)
        
        max_length = max(len(audio) for audio in x)

        # 填充音频数据并生成 mask
        padded_audio_list = []
        mask_list = []

        for audio in x:
            # 填充音频数据
            padded_audio = F.pad(torch.tensor(audio), (0, max_length - len(audio)))
            padded_audio_list.append(padded_audio)
            
            # 生成 mask
            mask = [1] * len(audio) + [0] * (max_length - len(audio))
            mask_list.append(mask)

        # 将列表转换为 PyTorch 张量
        padded_audio_tensor = torch.stack(padded_audio_list).unsqueeze(1)
        mask_tensor = torch.tensor(mask_list)
        
        return padded_audio_tensor.to(self.device).to(torch.bfloat16), mask_tensor.to(self.device).to(torch.bfloat16)
    
    def adjust_mask(self, mask, stride):
        # 假设 mask 的形状为 (batch_size, seq_len)
        batch_size, seq_len = mask.shape
        new_seq_len = (seq_len - 1) // stride + 1
        new_mask = mask[:, :new_seq_len * stride:stride]
        return new_mask

    def forward(self, x):
        input_dict = self.processor(
            x, return_tensors="pt", padding=True, sampling_rate=16000
        ).to(self.device,dtype=torch.bfloat16)
        
        mask = input_dict['attention_mask']
        mask = mask[:, :: (self.time_reduction_factor)]
        x = self.model(**input_dict).last_hidden_state
        

        mask = self.adjust_mask(mask, stride=2)
        x, mask = self.adapter(x, mask)#x:(B,T,hidden dim)
        # mask = torch.ones(x.shape[0],x.shape[1]).to(self.device,dtype=torch.bfloat16)
        
        
        assert mask.shape == x.shape[:2], f"Shape mismatch: mask.shape = {mask.shape}, x.shape[:2] = {x.shape[:2]}"
        return x, mask
