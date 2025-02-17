import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from .model import LORA_CONFIG
import re
import numpy as np

def my_save(args, trainer, dd, ff):
    if '14b-run1' in ff:
        fn = ff.split('/')[-1]
        fff = '/dev/shm/' + fn
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif ('world/14b' in ff) or ('world/7b' in ff):
        aa = ff.split('/')[1]
        fn = ff.split('/')[-1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True)
    else:
        torch.save(dd, ff)

from collections import deque

class Queue:
    def __init__(self, max_len=10):
        self.queue = deque(maxlen=max_len)
        self.sum = 0

    def enqueue(self, val):
        if len(self.queue) == self.queue.maxlen:
            self.sum -= self.queue[0]
        self.queue.append(val)
        self.sum += val

    def average(self):
        return self.sum / len(self.queue) if self.queue else None        

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.step = 0
        self.loss_queue = Queue(100)
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        if args.my_exit_tokens != 0: # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init                
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):
                    my_save(
                        args, trainer,
                        pl_module.state_dict(),
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                    exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_step} {lr}")

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        # param_name = {"speech_encoder.adapter.0.weight", "speech_encoder.adapter.0.bias", "speech_encoder.adapter.2.weight", "speech_encoder.adapter.2.bias"}
        
        # for name, param in pl_module.named_parameters():
        #     if name in param_name:
        #         print(f"{name} 的梯度: {param.grad}")
        
        self.step += 1
        if(self.step != 0 and self.step % 100 == 0 and trainer.is_global_zero):
            print("saving...")
            # to_save_dict = pl_module.state_dict()
            # names = ['layers.22', 'layers.23','layers.21','layers.20']
            filtered_state_dict = {}
            for key, param in pl_module.named_parameters():
                # 检查参数是否需要梯度
                if param.requires_grad:
                    # 添加需要梯度的参数到 filtered_state_dict
                    filtered_state_dict[key] = param.data
                elif('state' in key):
                    filtered_state_dict[key] = param.data
                # elif('speech_encoder' in key):
                #     filtered_state_dict[key] = param.data
                elif('lora' in key):
                    filtered_state_dict[key] = param.data
            
            # for key in pl_module.state_dict().keys():
            #     # Check if the key matches any of the commented weights
            #     if key.startswith('language_model.blocks.') and "att.time_state" in key:
            #         # Add the key and value to the filtered state dict
            #         filtered_state_dict[key] = pl_module.state_dict()[key]
            #     elif key.startswith('speech_encoder.adapter.'):
            #         filtered_state_dict[key] = pl_module.state_dict()[key]
                # for n in names:
                #     if(n in key):
                #         filtered_state_dict[key] = pl_module.state_dict()[key]
                # elif 
            
            try:
                import glob
                files = glob.glob(os.path.join(args.proj_dir, '*.pth'))
                for file in files:
                    os.remove(file)
                    
                my_save(
                    args, trainer,
                    filtered_state_dict,
                    f"{args.proj_dir}/rwkv-adapter-{self.step}.pth",
                )
            except Exception as e:
                print('Error\n\n', e, '\n\n')
        
        
        
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                # self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                # self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]
            else:
                # trainer.my_loss = trainer.my_loss_all.float().mean().item()
                # trainer.my_loss = trainer.my_loss_sum.float().mean().item()#修改
                trainer.my_loss = outputs["loss"]
            # trainer.my_loss_sum += trainer.my_loss
            # trainer.my_loss_count += 1
            # trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            try:
                self.loss_queue.enqueue(trainer.my_loss)
            except:
                print(f"misbehaved loss:{trainer.my_loss}")
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            # self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            self.log("loss", self.loss_queue.average(), prog_bar=True, on_step=True)
            self.log("step", trainer.my_loss, prog_bar=True, on_step=True)
            

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy): # save pth
            if args.magic_prime > 0:
                expand_factor = 2 if args.my_qa_mask > 0 else 1
                if int(real_step) == int(args.magic_prime * expand_factor // args.real_bsz) - 1 + int(args.my_random_steps):
                    to_save_dict = pl_module.state_dict()
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-final.pth",
                    )

        if args.LISA and (batch_idx+1)%args.lisa_k==0:
            pl_module.requires_grad_(False)
            select_layers = np.random.choice(range(args.n_layer), args.lisa_r, replace=False)
            
            for name, module in pl_module.named_modules():
                for pname, param in module.named_parameters():
                    if 'emb' in pname or 'head' in pname or '.ln' in pname or 'time' in pname:
                        param.requires_grad = True
                    elif 'ln_out' in pname:
                        param.requires_grad = True
                    match = re.search(r'\d+', pname)
                    if match:
                        number = int(match.group())
                        if number in select_layers:
                            param.requires_grad  = True
                break
        # if args.batch_save==batch_idx :
        #     to_save_dict = pl_module.state_dict()
        #     for name, state in to_save_dict.items():
        #         if 'img' in name:
        #             to_save_dict[name] = state
        #     try:
        #             my_save(
        #                 args, trainer,
        #                 to_save_dict,
        #                 f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}-{batch_idx}.pth",
        #             )
        #     except Exception as e:
        #         print('Error\n\n', e, '\n\n')
                

    def on_train_epoch_start(self, trainer, pl_module):

        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()

                if args.data_type=='img' and not args.lora:
                    for name, state in to_save_dict.items():
                        if 'img' in name:
                            to_save_dict[name] = state
                
                if args.state_tune or args.train_type=='state':
                    # lora_dict = {}
                    # for name, state in to_save_dict.items():
                    #     if 'state' in name:
                    #         lora_dict[name] = state
                    lora_dict = to_save_dict
                    to_save_dict = lora_dict


                if args.lora:
                    enable_time_finetune = 'time' in LORA_CONFIG["parts"]
                    enable_ln_finetune = 'ln' in LORA_CONFIG["parts"]
                    lora_dict = {}
                    for name, state in to_save_dict.items():
                        if len(args.load_model) == 0:
                            if 'emb' in name or 'head' in name or 'ln' in name:
                                lora_dict[name] = state
                        if args.emb and  'emb' in name:
                            lora_dict[name] = state
                        if ('.lora_' in name
                                or (enable_time_finetune and '.time_' in name)
                                or (enable_ln_finetune and '.ln' in name)):
                            lora_dict[name] = state
                    to_save_dict = lora_dict

                # try:
                #     import glob
                #     files = glob.glob(os.path.join(args.proj_dir, '*.pth'))
                #     for file in files:
                #         os.remove(file)
                        
                #     my_save(
                #         args, trainer,
                #         to_save_dict,
                #         f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                #     )
                # except Exception as e:
                #     print('Error\n\n', e, '\n\n')

        # if trainer.is_global_zero:  # logging
        #     trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
        #     trainer.my_log.flush()

        #     trainer.my_loss_sum = 0
        #     trainer.my_loss_count = 0
        #     if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
        #         exit(0)


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.my_pile_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.my_pile_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
