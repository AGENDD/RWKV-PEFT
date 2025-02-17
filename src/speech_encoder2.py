import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

from transformers import AutoProcessor, AutoModel
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer


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
        assert train_mode in ["adapter", "full", 'none']
        super(SpeechEncoder, self).__init__()

        self.device = device
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.time_reduction_factor = int(
            self.processor.feature_extractor.sampling_rate / 50
        )
        self.padding_length = 320
        # self.model = AutoModel.from_pretrained(model_id,cache_dir="temp_models").to(self.device,dtype=torch.bfloat16)
        # self.model = AutoModel.from_pretrained("microsoft/wavlm-large")
        
        self.model = AutoModel.from_pretrained(model_id,cache_dir="temp_models").to(self.device,dtype=torch.bfloat16)
        self.model.eval()
        self.model_output_dim = self.model.config.hidden_size
        self.downsample_K = downsample_K
        self.project_dim = project_dim
        if hidden_dim is None:
            self.hidden_dim = self.project_dim * 2
        else:
            self.hidden_dim = hidden_dim
        # adapter shall be a Linear(Relu(Linear)) structure
        self.adapter = nn.Sequential(
            nn.Linear(self.model_output_dim * self.downsample_K, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        ).to(self.device,dtype=torch.bfloat16)
        self.set_gradient(train_mode)

    def set_gradient(self, train_mode):
        """
        if train_mode is "adapter", only train the adapter layers, otherwise train the whole model
        """
        if train_mode == "adapter":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.adapter.parameters():
                param.requires_grad = True
            
            # names = ['layers.22', 'layers.23','layers.21','layers.20']
            # # names = ['layers.23']
            # # names = []
            # for name,param in self.model.named_parameters():
            #     for n in names:
            #         if(n in name):
            #             param.requires_grad = True
            
        elif train_mode == 'full':
            for param in self.model.parameters():
                param.requires_grad = True
            for param in self.adapter.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.adapter.parameters():
                param.requires_grad = False           

    def calculate_mask(self, input_dict):
        """
        Also need to handle the masking issue, to let the model not to attend to the padding tokens
        """
        attention_mask = input_dict["attention_mask"]  # [batch, num_samples]

        mask = attention_mask[:, :: (self.time_reduction_factor * self.downsample_K)]
        return mask

    def forward(self, x):
        input_dict = self.processor(
            x, return_tensors="pt", padding=True, sampling_rate=16000
        ).to(self.device,dtype=torch.bfloat16)
        mask = self.calculate_mask(input_dict)
        x = self.model(**input_dict).last_hidden_state
        # reshape the output from [batch_size, num_frames, hidden_size] to [batch_size, num_frames//downsample_K, hidden_size*downsample_K]
        x = x.unfold(1, self.downsample_K, self.downsample_K).flatten(2)
        x = self.adapter(x)
        mask = mask[:, : x.shape[1]]
        return x, mask
