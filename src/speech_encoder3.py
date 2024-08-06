import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

from transformers import AutoProcessor, AutoModel
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer

from speechtokenizer import SpeechTokenizer


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

        # feature_extractor = Wav2Vec2FeatureExtractor(
        #     feature_size=1,
        #     sampling_rate=16000,
        #     padding_value=0.0,
        #     do_normalize=True,
        #     return_attention_mask=False,
        # )
        self.device = device
        # try:
        #     self.processor = AutoProcessor.from_pretrained(model_id)
        # except:
        #     self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        # self.time_reduction_factor = int(
        #     self.processor.feature_extractor.sampling_rate / 50
        # )
        self.padding_length = 320
        
        config_path = "temp_models/ST/config.json"
        ckpt_path = "temp_models/ST/SpeechTokenizer.pt"
        
        
        self.model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).eval().to(self.device,dtype=torch.bfloat16)
        self.downsample_K = downsample_K
        
        # self.model_output_dim = self.model.config.hidden_size
        self.model_output_dim = self.model.n_q
        # self.downsample_K = downsample_K
        self.project_dim = project_dim
        if hidden_dim is None:
            self.hidden_dim = self.project_dim * 2
        else:
            self.hidden_dim = hidden_dim
            
        self.adapter = nn.Sequential(
            nn.Linear(self.model_output_dim * self.downsample_K, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        ).to(self.device,dtype=torch.bfloat16)
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

    def calculate_mask(self, input_dict):
        """
        Also need to handle the masking issue, to let the model not to attend to the padding tokens
        """
        attention_mask = input_dict["attention_mask"]  # [batch, num_samples]
        length_in_samples = (
            attention_mask.shape[1] // self.padding_length * self.padding_length
        )
        # calculate the mask length
        mask_length = length_in_samples // self.time_reduction_factor
        # create the mask
        mask = attention_mask[:, :: (self.time_reduction_factor * self.adapter.downsample_K)]
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
        

    def forward(self, x):
        # input_dict = self.processor(
        #     x, return_tensors="pt", padding=True, sampling_rate=16000
        # ).to(self.device,dtype=torch.bfloat16)
        x,mask = self.padding_mask(x)#x:(B,channel,T) mask:(B,T)
        
        x = self.model.encode(x)#x:(n_q,B,T)
        x = x.permute(1,2,0)#x:(B,T,n_q)
        x = x.unfold(1, self.downsample_K, self.downsample_K).flatten(2) #x:(B,T//k,n_q*k)
        
        
        # mask = self.calculate_mask(input_dict)
        # x = self.model(**input_dict).last_hidden_state
        # reshape the output from [batch_size, num_frames, hidden_size] to [batch_size, num_frames//downsample_K, hidden_size*downsample_K]
        # x = x.unfold(1, self.downsample_K, self.downsample_K).flatten(2)

        x = self.adapter(x)#x:(B,T,hidden dim)
        
        # mask = mask[:, : x.shape[1]]
        # mask = torch.ones(x.shape[0],x.shape[1])
        return x, mask
