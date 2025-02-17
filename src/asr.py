"""
The main body of the ASR model,

User: <Speech> <Prompt>
Model: <Transcription>
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List
from torch.nn.utils.rnn import pad_sequence

try:
    # from .speech_encoder import SpeechEncoder
    # from .speech_encoder2 import SpeechEncoder
    from .speech_encoder3 import SpeechEncoder
except ImportError:
    # from speech_encoder import SpeechEncoder
    # from speech_encoder2 import SpeechEncoder
    from speech_encoder3 import SpeechEncoder

from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import RWKV
# from .lora import LinearWithLoRA
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.strategies import DeepSpeedStrategy
import os, math, gc, importlib
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import time
import numpy as np
from contextlib import contextmanager, redirect_stdout, redirect_stderr
np.set_printoptions(threshold=np.inf)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random, librosa

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class SLAM_ASR(pl.LightningModule):
    def __init__(
        self,
        args,
        speech_encoder_model_id,
        language_model,
        downsample_K=5,
        hidden_dim=2048,
        # hidden_dim=4096,
        train_mode="adapter",
        device="cuda",
        token = "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",
    ):
        assert train_mode in ["adapter", "full", 'none']
        super().__init__()
        self.args = args
        self._device = device


        self.language_tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-6-world-1b6",trust_remote_code=True)
        # self.language_tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-6-world-1b6",trust_remote_code=True)

        self.language_model = language_model
        
        # for name, param in self.language_model.named_parameters():
        #     print(f"参数名称: {name}, 形状: {param.shape}")
        # exit(0)
        # language_project_dim = self.language_model.args.hidden_size
        # language_project_dim = 2560 3B
        # language_project_dim = 4096
        language_project_dim = args.n_embd
        
        self.speech_encoder = SpeechEncoder(
            speech_encoder_model_id,
            language_project_dim,
            downsample_K=downsample_K,
            hidden_dim=hidden_dim,
            train_mode=train_mode,
            device=device,
        ).to(self._device,torch.bfloat16)
      
        self.T_init = 0
        self.T_hubert = 0
        self.T_vector = 0
        self.T_rwkv = 0

        
        # for param in self.TTS.parameters():
        #     param.requires_grad = False
        self.set_gradient(train_mode,'state')

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
        assert train_mode in ["adapter", "full",'none']

        # call set_gradient for speech encoder
        self.speech_encoder.set_gradient(train_mode)
        
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
    
    # def audioAug(self,audio):

    #     audio = np.array(audio)
    #     sr = 16000
        
    #     ######################时域拉伸
    #     random_speed = random.uniform(0.7, 1.3)
        
    #     audio = librosa.effects.time_stretch(audio, rate = random_speed)
    #     # audio = audio.tolist()
            
    #     ######################音高变化
        
    #     n_steps = np.random.uniform(-4, 4)
    #     audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        
    #     ######################时域遮挡
        
    #     mask_duration = np.random.uniform(0, 0.2)
    #     mask_length = int(mask_duration * sr)
    #     mask_start = np.random.randint(0, len(audio) - mask_length)
    #     audio[mask_start:mask_start + mask_length] = 0
        
    #     ######################加噪
        
    #     noise_level = random_speed = random.uniform(0.0001, 0.001)
    #     noise = np.random.randn(len(audio))
    #     audio = audio + noise_level * noise
        
    #     audio = audio.tolist()
    #     return audio
    
    def _prepare_input_embeds(
        self, audios: List[float], transcriptions: List[str] = None
    ):
        """
        First, run audios through speech_encoder to get the embeddings and mask
        """
        # audios = [audio.cpu() for audio in audios]
        # print(f"audio:{len(audios)}-{[len(au) for au in audios]}")
        
        # for i in range(len(audios)):
        #     try:
        #         audios[i] = self.audioAug(audios[i])
        #     except:
        #         continue
        
        
        self.T_init = time.time()
        speech_output, mask = self.speech_encoder(audios)
        mask = mask.to(self._device)
        
        del audios
        # print(f"audio after hubert and adapter:\t{speech_output.shape}")
        # print(f"audio mask:\t{mask.shape}")
        self.T_audio = time.time()
        if transcriptions is not None:
            
            ###########处理prompt_embed ###############################################################################
            
            #去除speech padding
            audio_no_padding = self.remove_padding(speech_output,mask)  
            
            #在speech结尾添加end of audio：#
            end_of_audio = self.language_tokenizer(
                "#",
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                end_of_audio = self.language_model.embed(end_of_audio.input_ids)
            audio_no_padding_eoa = []
            for t in audio_no_padding:
                t = torch.cat((t, end_of_audio.squeeze(0)))
                audio_no_padding_eoa.append(t)
            
            #audio mask 左边添加1 (对应end of audio)
            ones = torch.ones(mask.size(0), 1).to(self._device)
            mask =torch.cat((ones, mask), dim=1)
            
            #处理transcription，得到embeded label
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
            
            #拼接speech和label
            audio_label = self.concatenate_audio_transcription(audio_no_padding_eoa , labels_embeds)
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
            
            # 剔除audio mask 右边的0
            mask_no_zero = []
            for mask_i in mask:
                mask_i_no_zero = mask_i[mask_i != 0]
                mask_no_zero.append(mask_i_no_zero)
            
            # 将audio mask和transcription mask 拼接
            mask_concatenate = self.concatenate_audio_transcription(mask_no_zero, att3)
            
            #向mask 填充0
            max_mask = max([len(x) for x in mask_concatenate])
            for i, x in enumerate(mask_concatenate):
                times = max_mask - len(x)
                for _ in range(times):
                    x = torch.cat((x,torch.tensor([0]).to(self.device)))
                mask_concatenate[i] = x

            #转换成tensor
            mask_concatenate = torch.stack(mask_concatenate)
            prompt_mask = mask_concatenate
            
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
            
            # 为transcription 结尾添加 end of sentence：<s>
            transcriptions_eos = []
            for starr in transcriptions:
                starr = starr + "<s>"
                transcriptions_eos.append(starr)
            _labels = self.language_tokenizer(
                transcriptions_eos,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(self.device)
            true_labels = _labels.input_ids

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
            true_labels = padded_labels
        else:           
            end_of_audio = self.language_tokenizer(
                "#",
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                end_of_audio = self.language_model.embed(end_of_audio.input_ids)
            #     print(f"end of audio:\n{end_of_audio}")
            
            # print(f"Speech output:\n{speech_output}")
            speech_output = torch.cat((speech_output, end_of_audio), dim= 1)
            
            prompt_embed = speech_output
            prompt_mask = mask
            true_labels = None
        return prompt_embed, prompt_mask, true_labels
    
    @contextmanager
    def suppress_stdout(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield

    def _prepare_input_tensor(self, tensors, transcriptions):
    ####################################### 建立 prompt_embed
        for i in range(len(tensors)):
            tensors[i] = torch.tensor(tensors[i]).to("cuda", torch.bfloat16)
        
        tensor_mask = [torch.zeros((tensor.shape[0],), dtype=int).to("cuda") for tensor in tensors]
        
        # 将 "#+transcription" 处理成 token
        transcriptions_with_eoa = ['#' + transcription for transcription in transcriptions]
        
        transcriptions_with_eoa_token = self.language_tokenizer(
            transcriptions_with_eoa,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self.device)
        
        filtered_tokens = [input_id[mask.bool()] for input_id, mask in zip(transcriptions_with_eoa_token.input_ids, transcriptions_with_eoa_token.attention_mask)]
        
        # 将 token 处理成 tensor
        transcriptions_with_eoa_embed = []
        with torch.no_grad():
            for tokens in filtered_tokens:
                transcriptions_with_eoa_embed.append(self.language_model.embed(tokens.unsqueeze(0)).squeeze(0))
            padding_embed = self.language_model.embed(torch.zeros((1, 1), dtype=torch.long).to("cuda"))[0].to(torch.bfloat16)
        
        # 拼接：audio tensor + transcript tensor
        # for i in range(len(tensors)):
        #     print(f"tensors{i}:{tensors.shape}")
        #     print(f"transcriptions_with_eoa_embed{i}:{transcriptions_with_eoa_embed.shape}")
        
        
        prompt_embed = [torch.cat([tensors[i], transcriptions_with_eoa_embed[i]], dim=0) for i in range(len(transcriptions))]
        
        max_length = max(len(tensor) for tensor in prompt_embed)
        
        # 使用 pad_sequence 进行填充
        prompt_embed = pad_sequence(
            [torch.cat([tensor, padding_embed[:max_length - len(tensor)]], dim=0) for tensor in prompt_embed],
            batch_first=True
        ).to("cuda")
        
        ############################################# 建立 label
        transcriptions_with_eos = [transcription + "<s>" for transcription in transcriptions]
        
        transcriptions_with_eos_token = self.language_tokenizer(
            transcriptions_with_eos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self.device)
        
        true_labels = [input_id[mask.bool()] for input_id, mask in zip(transcriptions_with_eos_token.input_ids, transcriptions_with_eos_token.attention_mask)]
        
        # label 左边填充 audio tensor 长度的 0
        for i in range(len(filtered_tokens)):
            true_labels[i] = torch.cat([tensor_mask[i], true_labels[i]], dim=0)
        
        max_length = max(len(tensor) for tensor in true_labels)
        
        # 使用 pad_sequence 进行填充
        true_labels = pad_sequence(
            [torch.cat([tensor, torch.zeros(max_length - len(tensor)).to(tensor.device)], dim=0) for tensor in true_labels],
            batch_first=True
        ).to("cuda")
        
        ################################################# prompt_mask
        attention_mask = transcriptions_with_eos_token.attention_mask
        attention_mask = [mask[:mask.nonzero()[-1] + 1] for mask in attention_mask]
        
        prompt_mask = [torch.cat([tensor_mask[i], attention_mask[i]]) for i in range(len(tensor_mask))]
        
        max_length = max(len(tensor) for tensor in prompt_mask)
        
        # 使用 pad_sequence 进行填充
        prompt_mask = pad_sequence(
            [torch.cat([tensor, torch.zeros(max_length - len(tensor)).to("cuda")], dim=0) for tensor in prompt_mask],
            batch_first=True
        )
        
        return prompt_embed, prompt_mask, true_labels.long()

    def output_split(self, outputs, labels, masks, transcriptions):
        end_of_asr = self.language_tokenizer(
            " ~",
            return_tensors="pt",
        ).to(self.device).input_ids.item()

        cut = []
        for i, l in enumerate(labels):
            indices = torch.where(l == end_of_asr)[0]
            if indices.numel() == 1:
                cut.append(indices.item())
            elif indices.numel() > 1:
                # 处理多个匹配的情况，例如只取第一个匹配的下标
                cut.append(indices[0].item())
            else:
                print(indices)
                print(l)
                print(transcriptions[i])
                exit(0)

        output1_list = []
        label1_list = []
        mask1_list = []
        output2_list = []
        label2_list = []
        mask2_list = []

        for i, c in enumerate(cut):
            o = outputs[i]
            l = labels[i]
            m = masks[i]

            o1 = o[:c+1]
            l1 = l[:c+1]
            m1 = m[:c+1]

            o2 = o[c+2:]
            l2 = l[c+2:]
            m2 = m[c+2:]

            output1_list.append(o1)
            label1_list.append(l1)
            mask1_list.append(m1)
            output2_list.append(o2)
            label2_list.append(l2)
            mask2_list.append(m2)

        output1 = torch.cat(output1_list, dim=0)
        label1 = torch.cat(label1_list, dim=0)
        mask1 = torch.cat(mask1_list, dim=0)
        output2 = torch.cat(output2_list, dim=0)
        label2 = torch.cat(label2_list, dim=0)
        mask2 = torch.cat(mask2_list, dim=0)

        # print(f"output1: {output1.shape}")
        # print(f"label1: {label1.shape}")
        # print(f"mask1: {mask1.shape}")
        # print(f"output2: {output2.shape}")
        # print(f"label2: {label2.shape}")
        # print(f"mask2: {mask2.shape}")

        return output1, label1, mask1, output2, label2, mask2
            
        
    def forward(self, audios: List[str], transcriptions: List[str] = None):
        
        # torch.cuda.empty_cache()
        
        # for i in range(len(audios)):
        #         print(f"{len(audios[i])/16000}:{len(transcriptions[i])}")
        # with torch.no_grad():
        prompt_embed, prompt_mask, true_labels = self._prepare_input_embeds(
            audios, transcriptions
        )
            
        # print(f"adapter output:{prompt_embed}")
        self.T_vector = time.time()
        outputs = self.language_model(inputs_embeds=prompt_embed)
        self.T_rwkv = time.time()
        
        # print("forward")
        mode = "qa"
        if(mode == 'qa'):
            output1, label1, mask1, output2, label2, mask2 = self.output_split(outputs, true_labels, prompt_mask, transcriptions)
            return outputs, true_labels, prompt_mask, output1, label1, mask1, output2, label2, mask2
        else:
            return outputs, true_labels, prompt_mask, None, None, None, None, None, None

        
    
    # def forward(self, tensors, transcriptions: List[str] = None):
        
    #     prompt_embed, prompt_mask, true_labels = self._prepare_input_tensor(
    #         tensors, transcriptions
    #     )

    #     self.T_vector = time.time()
    #     outputs = self.language_model(inputs_embeds=prompt_embed)
    #     self.T_rwkv = time.time()

    #     return outputs, true_labels, prompt_mask
    

    def generate(self,prompts: List[str] = None, audios: List[float] = None, tensor = None, endding='<s>', dy = False, length = 500):
        """
        Generate the transcription
        """
        
        if(audios is not None): #音频输入
            prompt_embed, prompt_mask, _ = self._prepare_input_embeds([audios])
        elif(prompts is not None): #文本输入
            prompts_tokens = self.language_tokenizer(
                [prompts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(self.device)
        
            with torch.no_grad():
                # labels_embeds = self.language_model.rwkv.get_input_embeddings()(_labels.input_ids)
                prompt_embed = self.language_model.embed(prompts_tokens.input_ids)
                prompt_mask = prompts_tokens.attention_mask
        elif(tensor != None): #向量输入
            
            prompt_embed = tensor.unsqueeze(0)
            
            end_of_audio = self.language_tokenizer(
                "#",
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                end_of_audio = self.language_model.embed(end_of_audio.input_ids)
            #     print(f"end of audio:\n{end_of_audio}")
            
            # print(f"Speech output:\n{speech_output}")
            prompt_embed = torch.cat((prompt_embed, end_of_audio), dim= 1)
            
            
        self.language_model.to(self._device, dtype=torch.bfloat16)
        outputs = self.language_model.generate(tokenizer= self.language_tokenizer,inputs_embeds=prompt_embed, endding=endding, dy = dy, LENGTH = length)
        
        return outputs

    def training_step(self, batch, batch_idx):
            args = self.args
            if args.loss_mask:
                idx, targets, mask = batch
                mask = mask.view(-1)
                sum_mask = torch.sum(mask).item()
                logits = self(idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                loss = torch.sum(loss * mask) / sum_mask
            # elif args.my_qa_mask != 1:
            #     idx, targets = batch
            #     logits = self(idx)
            #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # if '0' in os.environ["RWKV_MY_TESTING"]:
                #     print('logits', logits)
                #     torch.set_printoptions(threshold=10000)
                #     print('idx', idx)
                #     exit(0)
            else:
                
                ##改动
                # idx, transcription = batch
                # print(f"batch length:{len(batch)}")
                # print(f"batch item:{len(batch[0])}")
                # print(f"batch item[0]:{len(batch[0][0])}:{type(batch[0][0])}")
                # print(f"batch item[1]:{len(batch[0][1])}:{type(batch[0][1])}")
                idx = [item[0] for item in batch]
                transcription = [item[1] for item in batch]
                # logits, targets, mask = self(idx, transcription)
                #mask = mask.reshape(-1)
                # sum_mask = torch.sum(mask).item()
                
                # print(type(idx[0]))
                
                logits, targets, mask,logits1, targets1, mask1,logits2, targets2, mask2 = self(idx, transcription)
                
                if(logits1 is not None):
                    sum_mask1 = torch.sum(mask1).item()
                    sum_mask2 = torch.sum(mask2).item()

                    try:
                        # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none')
                        # loss_raw = loss
                        # loss = torch.sum(loss * mask) / sum_mask
                        
                        loss1 = F.cross_entropy(logits1, targets1, reduction='none')               
                        loss2 = F.cross_entropy(logits2, targets2, reduction='none')
                        
                        loss1 = torch.sum(loss1 * mask1) / sum_mask1
                        loss2 = torch.sum(loss2 * mask2) / sum_mask2
                        
                        loss = 0.7*loss1 + 0.3*loss2
                        
                    except:
                        loss = torch.tensor([0.0], device=logits.device)  
                        print("zero loss")
                else:
                    mask = mask.reshape(-1)
                    sum_mask = torch.sum(mask).item()
                    
                    try:

                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none')
                        loss = torch.sum(loss * mask) / sum_mask

                        
                    except:
                        loss = torch.tensor([0.0], device=logits.device)  
                        print("zero loss")
                    
            return L2Wrap.apply(loss, logits)
    
    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    def return_tokenizer(self):
        return self.language_tokenizer
    
    @property
    def config(self):
        return self.language_model.config
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        
        self._device = value


    
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
