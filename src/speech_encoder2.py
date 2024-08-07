import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

from transformers import AutoProcessor, AutoModel
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer


class Adapter(nn.Module):
    def __init__(self, model_output_dim, project_dim):
        super(Adapter, self).__init__()
        # 一维卷积层，步长为2，核大小为3
        self.conv1 = nn.Conv1d(model_output_dim, model_output_dim*2, kernel_size=3, stride=2, padding=1)
        # 添加一个下采样操作，这里我们使用最大池化
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(model_output_dim*2, model_output_dim*3, kernel_size=3, stride=2, padding=1)
        # 添加一个下采样操作，这里我们使用最大池化
        self.pool2 = nn.MaxPool1d(2)
        
        # Transformer层，latent dimension为3072
        encoder_layers = TransformerEncoderLayer(d_model=model_output_dim*3, nhead=8, dim_feedforward=3072)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=2)
        # 前馈层，维度为4096
        self.ffn = nn.Sequential(
            nn.Linear(model_output_dim*3, 4096),
            nn.ReLU(),
        )
        # 线性层，输出和原始输出有相同的形状
        self.linear = nn.Linear(4096, project_dim)

    def forward(self, x):
        # 交换维度，使得卷积在seq length维度上进行:(batch, feature, seq_length)
        x = x.transpose(1, 2)
        # 对输入进行一维卷积
        x = self.conv1(x)
        # 对卷积的输出进行下采样
        x = self.pool1(x)
        # 对输入进行一维卷积
        x = self.conv2(x)
        # 对卷积的输出进行下采样
        x = self.pool2(x)
        # 再次交换维度，使得输出的形状与原始输入的形状相同
        x = x.transpose(1, 2)
        # 再次交换维度，使得输出的形状为:(seq_length, batch, feature)
        x = x.transpose(0, 1)
        # 将输出送入Transformer层
        x = self.transformer(x)
        # 变回:(batch, seq_length, feature)
        x = x.transpose(0, 1)
        # 将输出送入前馈层
        x = self.ffn(x)
        # 将输出送入线性层，得到最终输出
        x = self.linear(x)
        return x


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

        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        self.device = device
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except:
            self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.time_reduction_factor = int(
            self.processor.feature_extractor.sampling_rate / 50
        )
        self.padding_length = 320
        self.model = AutoModel.from_pretrained(model_id).to(self.device,dtype=torch.bfloat16)
        self.model_output_dim = self.model.config.hidden_size
        # self.downsample_K = downsample_K
        self.project_dim = project_dim
        if hidden_dim is None:
            self.hidden_dim = self.project_dim * 2
        else:
            self.hidden_dim = hidden_dim
        # adapter shall be a Linear(Relu(Linear)) structure
        # self.adapter = nn.Sequential(
        #     nn.Linear(self.model_output_dim * self.downsample_K, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.project_dim),
        # ).to(self.device,dtype=torch.bfloat16)
        self.adapter = Adapter(self.model_output_dim, self.project_dim).to(self.device,dtype=torch.bfloat16)
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
        # mask = attention_mask[:, :: (self.time_reduction_factor * self.downsample_K)]
        mask = attention_mask[:, :: (self.time_reduction_factor * 2)]
        return mask

    def forward(self, x):
        input_dict = self.processor(
            x, return_tensors="pt", padding=True, sampling_rate=16000
        ).to(self.device,dtype=torch.bfloat16)
        mask = self.calculate_mask(input_dict)
        x = self.model(**input_dict).last_hidden_state
        # reshape the output from [batch_size, num_frames, hidden_size] to [batch_size, num_frames//downsample_K, hidden_size*downsample_K]
        # x = x.unfold(1, self.downsample_K, self.downsample_K).flatten(2)
        mm = input_dict["attention_mask"]

        x = self.adapter(x)
        
        # mask = mask[:, : x.shape[1]]
        mask = torch.ones(x.shape[0],x.shape[1]).to(self.device)
        return x, mask
