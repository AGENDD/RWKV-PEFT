"""
The main body of the ASR model,

User: <Speech> <Prompt>
Model: <Transcription>
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List

try:
    from .speech_encoder import SpeechEncoder
except ImportError:
    from speech_encoder import SpeechEncoder

from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import RWKV
from .lora import LinearWithLoRA


class SLAM_ASR(nn.Module):
    def __init__(
        self,
        args,
        speech_encoder_model_id,#facebook/hubert-base-ls960
        language_model,
        downsample_K=5,
        hidden_dim=2048,
        train_mode="adapter",
        device="cuda",
        token = "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",
    ):
        assert train_mode in ["adapter", "full"]
        super(SLAM_ASR, self).__init__()
        self.device = device
        """
                       |------|
                       |hubert|
                       |------|
                           |
                           |
                       (adapter, the only trainable part here)
                           |
                           v
        LLAMA("<user>: [______], transcribe it, <assistant>") --> "The weather is good"

        """

        # self.language_tokenizer = AutoTokenizer.from_pretrained(language_model_id,trust_remote_code=True)
        # self.language_model = AutoModelForCausalLM.from_pretrained(
        #     language_model_id,
        #     trust_remote_code=True,
        #     # token = token
        # ).to(self.device)
        # self.language_model = AutoModelForCausalLM.from_pretrained("temp_models/rwkv-6-world-1b6", trust_remote_code=True)
        
        # self.language_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.language_model.resize_token_embeddings(len(self.language_tokenizer))
        
        ###
        # print("Model before lora:")
        # print(self.language_model)
        # self.load_lora(self.language_model)
        # print("Model after lora:")
        # print(self.language_model)
        ###
        
        ########################################换成RWKV-PEFRT的模型结构

        self.language_model = language_model
        #########################################
                
        # language_project_dim = self.language_model.args.hidden_size
        language_project_dim = 2048
        
        self.speech_encoder = SpeechEncoder(
            speech_encoder_model_id,
            language_project_dim,
            downsample_K=downsample_K,
            hidden_dim=hidden_dim,
            train_mode=train_mode,
            device=device,
        ).to(self.device)
      
        
        # print("show language params")
        # for name,param in self.language_model.named_parameters():
        #     print(f"{name}:{param.requires_grad}")
        # print("show language model")
        # print(self.language_model)


        self.set_gradient(train_mode,'state')

        # self.prompt_part1 = """User:"""
        # self.prompt_part2 = (
        #     """transcribe it.\nAssistant:"""
        # )
        # self.embed_bank = {"embed1": None, "embed2": None, "att1": None, "att2": None}
        # self.set_embed_bank()

    def gradient_checkpointing_enable(self, **kwargs):
        self.language_model.gradient_checkpointing_enable(**kwargs)
                
    def load_lora(self, model):
        to_replace = []
        for name, child in model.named_children():
            if isinstance(child, nn.Linear):
                to_replace.append((name, child))
            else:
                self.load_lora(child)
        for name, child in to_replace:
            new_layer = LinearWithLoRA(child, 128, self.device)
            # new_layer.print_parameters()
            delattr(model, name)
            model.add_module(name, new_layer)
        for param in model.parameters():
            pass
    
    def set_embed_bank(self, batch_size=1):
        input_dict1 = self.language_tokenizer(
            [self.prompt_part1], return_tensors="pt"
        ).to(self.device)
        input_dict2 = self.language_tokenizer(
            [self.prompt_part2], return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        # precache the embeddings for prompt
        with torch.no_grad():
            inputs_embeds1 = self.language_model.rwkv.get_input_embeddings()(
                input_dict1.input_ids
            )
            inputs_embeds2 = self.language_model.rwkv.get_input_embeddings()(
                input_dict2.input_ids
            )
        self.embed_bank["embed1"] = inputs_embeds1
        self.embed_bank["embed2"] = inputs_embeds2
        self.embed_bank["att1"] = input_dict1.attention_mask
        self.embed_bank["att2"] = input_dict2.attention_mask
        print("[Preloaded embeddings for both part of the prompt.]")
        print(
            f"    {self.prompt_part1}        {inputs_embeds1.shape}\n    {self.prompt_part2}        {inputs_embeds2.shape}"
        )

    def set_gradient(self, train_mode,tuning):
        assert train_mode in ["adapter", "full"]

        # call set_gradient for speech encoder
        self.speech_encoder.set_gradient(train_mode)

        # freeze the whole language_model
        # if train_mode != "full":
        #     for name, param in self.language_model.named_parameters():
        #         # print(f"layer:{name}")
                
        #         if(tuning == 'lora' and 'lora' not in name.lower()):
        #             param.requires_grad = False
        #         elif(tuning == 'state'):
        #             if('state' not in name.lower()):
        #                 param.requires_grad = False
        #             else:
        #                 param.requires_grad = True
                
            
        # now list all parameters that require grad
        print("Parameters that require grad:")

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"    {name}: {param.shape}")

    
    def remove_padding(self, x, mask):
        #根据mask去除speech_output的padding部分
        x_no_padding = []
        # 对于每一个样本和对应的掩码
        for x_i, mask_i in zip(x, mask):
            # 使用掩码来选择非填充部分
            x_i_no_padding = x_i[mask_i.bool()]
            # 将结果添加到列表中
            x_no_padding.append(x_i_no_padding)
        
        return x_no_padding
    
    def concatenate_audio_transcription(self, audio, transcription):
        #将两个二维/三维向量在第二维度拼起来
        result = []
        for sublist1, sublist2 in zip(audio, transcription):
            sub_result = torch.cat((sublist1 ,sublist2), dim=0)
            result.append(sub_result)

        return result
    
    
    def _prepare_input_embeds(
        self, audios: List[float], transcriptions: List[str] = None
    ):
        """
        First, run audios through speech_encoder to get the embeddings and mask
        """
        
        speech_output, mask = self.speech_encoder(audios)
        # print(f"audio after hubert and adapter:\t{speech_output.shape}")
        # print(f"audio mask:\t{mask.shape}")
        
        if transcriptions is not None:
            
            ###########处理prompt_embed ###############################################################################
                #去除speech padding
            audio_no_padding = self.remove_padding(speech_output,mask)
            # print(f"audio with no padding:\t{len(audio_no_padding)}-{[len(x) for x in audio_no_padding]}")
            
            _labels = self.language_tokenizer(
                transcriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad():
                # labels_embeds = self.language_model.rwkv.get_input_embeddings()(_labels.input_ids)
                labels_embeds = self.language_model.embed(_labels.input_ids)
            att3 = _labels.attention_mask
            # print(f"embed transcription:\t{labels_embeds.shape}")
            # print(f"transcription mask:\t{att3.shape}")
            
                #拼接speech和label
            audio_label = self.concatenate_audio_transcription(audio_no_padding , labels_embeds)
            # print(f"concatenated inputs:\t{len(audio_label)}-{[len(x) for x in audio_label]}")
        
                #对拼接后的内容进行padding
            max_seq = max([len(x) for x in audio_label])
            for i, x in enumerate(audio_label):
                times = max_seq - len(x)
                for _ in range(times):
                    x = torch.cat((x,x[len(x)-1].unsqueeze(0)))
                audio_label[i] = x
            # print(f"padded inputs:\t{len(audio_label)}-{[len(x) for x in audio_label]}")
            
                #转换成tensor
            audio_label = torch.stack(audio_label)
            # print(f"padded inputs tensor:\t{audio_label.shape}")
            prompt_embed = audio_label
            # print()
            
            #####处理prompt_mask ##################################################
            
                # 剔除audio mask的0
            mask_no_zero = []
            for mask_i in mask:
                mask_i_no_zero = mask_i[mask_i != 0]
                mask_no_zero.append(mask_i_no_zero)
            # print(f"audio mask with no 0:\t{len(mask_no_zero)}-{[len(x) for x in mask_no_zero]}")
            
                # 将audio mask和transcription mask 拼接
            mask_concatenate = self.concatenate_audio_transcription(mask_no_zero, att3)
            # print(f"audio mask + transcription mask:\t{len(mask_concatenate)}-{[len(x) for x in mask_concatenate]}")
            
                #向mask 填充0
            max_mask = max([len(x) for x in mask_concatenate])
            for i, x in enumerate(mask_concatenate):
                times = max_mask - len(x)
                for _ in range(times):
                    x = torch.cat((x,torch.tensor([0]).to(self.device)))
                mask_concatenate[i] = x
            # print(f"padded inputs:\t{len(mask_concatenate)}-{[len(x) for x in mask_concatenate]}")
            
                #转换成tensor
            mask_concatenate = torch.stack(mask_concatenate)
            # print(f"padded attention mask tensor:\t{mask_concatenate.shape}")
            prompt_mask = mask_concatenate
            # print()
            
            # #########处理loss mask #####################################################
            # import torch.nn.functional as F
            # loss_mask = []
            
            # for t in mask_no_zero:
            #     pad_len = max_mask - len(t)
            #     pad = F.pad(t, (0, pad_len), "constant", 0)
            #     loss_mask.append(pad)
            
            # loss_mask = torch.stack(loss_mask)
            # loss_mask = prompt_mask - loss_mask
            
            # print(f"loss mask:\t{loss_mask.shape}")
            
            #########处理true_labels ###################################################
            # print()
            true_labels = _labels.input_ids
            # print(f"labels:\t{true_labels.shape}")
            
                #在ture label左侧填充audio 长度的-100， 同时在右侧填充-100使batch对齐
            padded_labels = []
            for i,t in enumerate(true_labels):
                back_padding = max_mask - t.shape[0] - audio_no_padding[i].shape[0]
                t = torch.cat(
                    [
                        torch.full(
                            (audio_no_padding[i].shape[0], ),
                            -100,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        t,
                        torch.full(
                            (back_padding, ),
                            -100,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ]
                )
                padded_labels.append(t)
            
            padded_labels = torch.stack(padded_labels)
            # print(f"true_labels:\t{padded_labels.shape}")
            true_labels = padded_labels
        else:
            prompt_embed = speech_output
            prompt_mask = mask
            true_labels = None
        return prompt_embed, prompt_mask, true_labels

    def forward(self, audios: List[float], transcriptions: List[str] = None):
        
        prompt_embed, prompt_mask, true_labels = self._prepare_input_embeds(
            audios, transcriptions
        )
        # run the prompt through the language model

        # print()
        # print(f"prompt_embed:{prompt_embed.shape}")
        # print(f"attention_mask:{prompt_mask.shape}")
        # print(f"true_labels:{true_labels.shape}")
        
        # outputs = self.language_model(
        #     inputs_embeds=prompt_embed,
        #     attention_mask=prompt_mask.bool(),
        #     labels=true_labels,
        # )  # CausalLMOutputWithPast
        
        outputs = self.language_model(inputs_embeds=prompt_embed)
        
        
        # print(f"outputs:{outputs['loss']}")
        # print(f"logits:\t{outputs['logits'].shape}")
        
        return outputs, true_labels
    

    def generate(self, audios: List[float], stopping_criteria=None):
        """
        Generate the transcription
        """
        prompt_embed, prompt_mask, _ = self._prepare_input_embeds(audios)
        
        outputs = self.language_model(
            inputs_embeds=prompt_embed,
            attention_mask=prompt_mask.bool()
        )
        return outputs

    @property
    def config(self):
        return self.language_model.config
