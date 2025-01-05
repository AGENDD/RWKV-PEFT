########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()
    
    parser.add_argument("--OP", default="1", type=int)

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--weight_decay_final", default=-1, type=float)

    parser.add_argument("--my_pile_version", default=1, type=int)  # my special pile version
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='x052', type=str)
    parser.add_argument("--my_exit", default=99999999, type=int)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    #LORA
    parser.add_argument("--emb", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_load", default="", type=str)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=32, type=float)
    parser.add_argument("--lora_dropout", default=0.01, type=float)
    parser.add_argument("--lora_parts", default="att,ln,time", type=str)

    #LISA
    parser.add_argument("--LISA", action="store_true")
    parser.add_argument("--lisa_r", default=2, type=int)
    parser.add_argument("--lisa_k", default=100, type=int)

    #PISSA
    parser.add_argument("--PISSA", action="store_true")
    parser.add_argument("--svd_niter", default=4, type=int)
    parser.add_argument("--pissa_load", default="", type=str)
    parser.add_argument("--pissa_init", default="", type=str)

    #quant
    parser.add_argument("--quant", default="none", type=str)

    #dataset
    parser.add_argument("--dataload", default="get", type=str)

    #state tuning
    parser.add_argument("--state_tune", action="store_true")


    parser.add_argument("--chunk_ctx", default=512, type=int)
    #fla
    parser.add_argument("--fla", action="store_true")
    parser.add_argument("--train_type", default="none", type=str)

    #loss_mask
    parser.add_argument("--loss_mask", action="store_true")

    if pl.__version__[0]=='2':
        parser.add_argument("--accelerator", default="gpu", type=str)
        parser.add_argument("--strategy", default="auto", type=str)
        parser.add_argument("--devices", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--precision", default="fp16", type=str)
        parser.add_argument("--accumulate_grad_batches", default=4, type=int)
    else:
        parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 10.0
    args.gradient_accumulation_steps = 4
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    if args.dataload!='get':
        args.max_epochs = args.epoch_count
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    ######state tuning
    os.environ["RWKV_TRAIN_TYPE"]=''
    if args.train_type=='state':
        os.environ["RWKV_TRAIN_TYPE"]='states'
    elif args.train_type=='infctx':
        os.environ["RWKV_TRAIN_TYPE"]='infctx'

    os.environ["WKV"]='fla' if args.fla else ''
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    if args.data_type == "wds_img":
        args.run_name = f"v{args.my_img_version}-{args.my_img_size}-{args.my_img_bit}bit-{args.my_img_clip}x{args.my_img_clip_scale}"
        args.proj_dir = f"{args.proj_dir}-{args.run_name}"
    else:
        args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    if args.my_pile_stage > 0:
        magic_prime_bak = args.magic_prime

        if args.my_pile_shift < 0:
            args.my_pile_shift = 0

        if magic_prime_bak > 0:
            args.magic_prime = magic_prime_bak
        if args.my_qa_mask == 2:
            args.epoch_count = 2 * args.magic_prime // 40320
        else:
            args.epoch_count = args.magic_prime // 40320

        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320
        # if args.my_pile_stage == 2:
        #     assert args.lr_final == args.lr_init
        if args.my_pile_stage >= 2:  # find latest saved model
            list_p = []
            for p in os.listdir(args.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    p = ((p.split("-"))[1].split("."))[0]
                    if p != "final":
                        if p == "init":
                            p = -1
                        else:
                            p = int(p)
                        list_p += [p]
            list_p.sort()
            max_p = list_p[-1]
            if len(list_p) > 1:
                args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if args.warmup_steps < 0:
                    if args.my_pile_stage == 2:
                        args.warmup_steps = 10
                    else:
                        args.warmup_steps = 30
            args.epoch_begin = max_p + 1

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-5 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
# Found deepspeed {deepspeed_version}, recommend 0.7.0 (faster than newer versions)
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "0"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    from src.dataset2 import MyDataset

    # train_data = MyDataset(args)
    # args.vocab_size = train_data.vocab_size
    from src.rwkvLinear import LORA_CONFIG, LoraLinear
    from src.model import RWKV
    if args.lora:
        assert args.lora_r > 0, "LoRA should have its `r` > 0"
        LORA_CONFIG["r"] = args.lora_r
        LORA_CONFIG["alpha"] = args.lora_alpha
        LORA_CONFIG["dropout"] = args.lora_dropout
        LORA_CONFIG["parts"] = set(str(args.lora_parts).split(','))
        enable_time_finetune = 'time' in LORA_CONFIG["parts"]
        enable_ln_finetune = 'ln' in LORA_CONFIG["parts"]
    if args.quant!='none':
        LORA_CONFIG["quant"]=True
    model = RWKV(args)
    freeze=False
    if args.lora or args.LISA or args.train_type=='state':
        model.requires_grad_(False)
        freeze=True
    
    if args.state_tune or args.train_type=='state':
        for name, module in model.named_modules():
            for pname, param in module.named_parameters():
                if 'state' in pname :
                    param.requires_grad = True
            break
    elif args.LISA:
        import re
        select_layers = np.random.choice(range(args.n_layer), args.lisa_r, replace=False)
        for name, module in model.named_modules():
            for pname, param in module.named_parameters():
                if 'emb' in pname or 'head' in pname or '.ln' in pname or 'time' in pname :
                    param.requires_grad = True
                match = re.search(r'\d+', pname)
                if match:
                    number = int(match.group())
                    if number in select_layers:
                        param.requires_grad  = True
            break
    elif args.lora:
        for name, module in model.named_modules():
            if len(args.load_model) == 0:
                if any(n.startswith("emb.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if 'emb.weight'==pname:
                            print(f'  EMB additionally training module {pname}')
                            param.requires_grad = True
                if any(n.startswith("head.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if 'head.weight'==pname:
                            print(f'  head additionally training module {pname}')
                            param.requires_grad = True
                if 'ln' in name:
                    print(f'  LoRA additionally training module {name}')
                    for param in module.parameters():
                        param.requires_grad = True
            if any(n.startswith("emb.") for n, _ in module.named_parameters()):
                for pname, param in module.named_parameters():
                    if args.emb and 'emb.weight'==pname:
                        print(f'  EMB additionally training module {pname}')
                        param.requires_grad = True
            if any(n.startswith("head.") for n, _ in module.named_parameters()):
                for pname, param in module.named_parameters():
                    if args.emb and 'head.weight'==pname:
                        print(f'  head additionally training module {pname}')
                        param.requires_grad = True
            if any(n.startswith("lora_") for n, _ in module.named_parameters()):
                print(f'  LoRA additionally training module {name}')
                for pname, param in module.named_parameters():
                    param.requires_grad = 'lora_' in pname
            elif enable_ln_finetune and '.ln' in name:
                print(f'  LoRA additionally training module {name}')
                for param in module.parameters():
                    param.requires_grad = True
            elif enable_time_finetune and any(n.startswith("time") for n, _ in module.named_parameters()):
                for pname, param in module.named_parameters():
                    if pname.startswith("time"):
                        print(f'  LoRA additionally training parameter {pname}')
                        param.requires_grad = True

    if len(args.load_model) == 0 or args.my_pile_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    # try:
    #     load_dict = torch.load(args.load_model, map_location="cpu")
    #     load_keys = list(load_dict.keys())
    #     for k in load_keys:
    #         if k.startswith('_forward_module.'):
    #             assert 1==2
    #             load_dict[k.replace('_forward_module.','')] = load_dict[k]
    #             del load_dict[k]
    # except:
    #     rank_zero_info(f"Bad checkpoint {args.load_model}")
    #     if args.my_pile_stage >= 2:  # try again using another checkpoint
    #         max_p = args.my_pile_prev_p
    #         if max_p == -1:
    #             args.load_model = f"{args.proj_dir}/rwkv-init.pth"
    #         else:
    #             args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
    #         args.epoch_begin = max_p + 1
    #         rank_zero_info(f"Trying {args.load_model}")
    #         load_dict = torch.load(args.load_model, map_location="cpu")

    # if args.load_partial == 1:
    #     load_keys = load_dict.keys()
    #     for k in model.state_dict():
    #         if k not in load_keys:
    #             load_dict[k] = model.state_dict()[k]
    model.load_state_dict(torch.load(args.load_model, map_location="cpu"), strict=(not freeze))

    if args.PISSA and args.pissa_init=="":
        init_dict = {}
        rank_zero_info(f"########## Init PISSA... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "pissa_init") and callable(getattr(m, "pissa_init")):
                m.pissa_init(args.svd_niter)
                init_dict[f'{name}.init_lora_A'] = m.lora_A.data
                init_dict[f'{name}.init_lora_B'] = m.lora_B.data
        torch.save(init_dict, f'{args.proj_dir}/init_pissa.pth')
    if os.path.isfile(args.lora_load):
        model.load_state_dict(torch.load(args.lora_load, map_location="cpu"),
                              strict=False)
        
    if os.path.isfile(args.pissa_load):
        model.load_state_dict(torch.load(args.pissa_load, map_location="cpu"),
                            strict=False)
        pissa_init = torch.load(args.pissa_init, map_location="cpu")
        rank_zero_info(f"########## Load PISSA... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "pissa_load") and callable(getattr(m, "pissa_load")):
                m.pissa_load(pissa_init[f'{name}.init_lora_A'], pissa_init[f'{name}.init_lora_B'])
    
    if args.quant!='none':
        rank_zero_info(f"########## Quant... ##########")
        for name, m in model.named_modules():
            if hasattr(m, "quant") and callable(getattr(m, "quant")):
                    m.quant(args.quant)


    if pl.__version__[0]=='2':
        trainer = Trainer(accelerator=args.accelerator,strategy=args.strategy,devices=args.devices,num_nodes=args.num_nodes,precision=args.precision,
        logger=args.logger,callbacks=[train_callback(args)],max_epochs=args.max_epochs,check_val_every_n_epoch=args.check_val_every_n_epoch,num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,enable_checkpointing=args.enable_checkpointing,accumulate_grad_batches=args.accumulate_grad_batches,gradient_clip_val=args.gradient_clip_val)
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[train_callback(args)],
        )

    if trainer.global_rank == 100:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 1:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
            else:
                print(f"{str(shape[0]).ljust(5)}       {n}")

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    # data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
    # model = model.to(dtype=torch.bfloat16)
    
    
    # print(model)
    
    # # 或者只打印参数名称和形状
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Parameter shape: {param.shape}")
    

    #########################LORA#############################################
    
    import torch
    import torch.nn as nn

    class LoRALayer(nn.Module):
        def __init__(self, in_features, out_features, r=64):
            super(LoRALayer, self).__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.lora_A = nn.Parameter(torch.randn(in_features, r))
            self.lora_B = nn.Parameter(torch.randn(r, out_features))
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
            self.r = r

        def forward(self, x):
            return self.linear(x) + (x @ self.lora_A @ self.lora_B)
    
    def replace_linear_with_lora(model, r=64, flag = False):
        
        def change(model_):
            for name, module in model_.named_children():
                if isinstance(module, nn.Linear):
                    print(f"\t{name}")
                    in_features = module.in_features
                    out_features = module.out_features
                    lora_layer = LoRALayer(in_features, out_features, r)
                    lora_layer.linear.weight = module.weight
                    lora_layer.linear.bias = module.bias
                    setattr(module, name, lora_layer)
        
        blocks = model.get_submodule("blocks")
        for name, module in blocks.named_children():
            
            if(int(name) < 3):
                print(name)
                att = module.get_submodule("att")
                ffn = module.get_submodule("ffn")
                change(att)
                change(ffn)
        return model
    
    # print("Change to LORA:")
    # model = replace_linear_with_lora(model, r=1024)
    
    # for name, param in model.named_parameters():
    #     if 'state' in name:
    #         param.requires_grad = False
    # print(model)
    # print("Paramter that require grad:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter name: {name}, Parameter shape: {param.shape}")
    
    ###########################################################################
    
    from src.asr import SLAM_ASR
    Total_model = SLAM_ASR(
        args,
        # "facebook/hubert-large-ls960-ft", # SHOULD NOT BE USED, THIS IS A FINETUNED VERSION.
        # "microsoft/wavlm-base-plus",
        "microsoft/wavlm-large",
        # "jonatasgrosman/exp_w2v2t_zh-cn_wavlm_s368",
        # "facebook/hubert-large-ll60k",
        model,
        # downsample_K=1,
    )
    print("loading weights...")
    import glob
    file_paths = glob.glob('output/rwkv-adapter*.pth')
    # file_paths = glob.glob('output/rwkv*.pth')
    # 检查是否找到了文件
    if file_paths:
        file_path = file_paths[0]
        model_state_dict = torch.load(file_path)
        Total_model.load_state_dict(model_state_dict, strict=False)
        print(f"Loaded model from {file_path}")
        # 打印所有加载的参数名
        for name, param in model_state_dict.items():
            print("\t"+name)
    else:
        print("No files found. Loading origin model.")
    
    
    Total_model = Total_model.to(dtype=torch.bfloat16)

    token = "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"
    from datasets import load_from_disk,load_dataset, concatenate_datasets
    

    def aishell(split="train"):
        if(split == 'train'):
            train_path = "temp_datasets/aishell/data_aishell/wav/train"
        else:
            train_path = "temp_datasets/aishell/data_aishell/wav/test"
        
        
        wav_filenames = []
        for filename in os.listdir(train_path):
            if filename.endswith('.wav'):
                wav_filenames.append(os.path.splitext(filename)[0])
        
        
        trans_path = 'temp_datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt'
        result_dict = {}
        with open(trans_path, 'r') as file:
            for line in file:
                elements = line.strip().split(' ')
                key = elements[0]
                value = ' '.join(elements[1:])
                result_dict[key] = value
        
        print(len(wav_filenames))
        print(len(result_dict))
        
        dataset_final = []
        for i in wav_filenames:
            if(i not in result_dict.keys()):
                print(f"{i} not found")
            else:
                dataset_final.append(i)
        print(len(dataset_final))
        return dataset_final,result_dict
    
    import librosa
    import resampy
    import scipy.io.wavfile as wav
    import re
    
    if(args.OP == 1):
        # dataset = load_dataset('covost2','zh-CN_en',data_dir = 'temp_datasets/covost-zhCN_en')['train']
        # dataset2 = load_dataset('covost2','ja_en',data_dir = 'temp_datasets/covost-ja_en')['train']
        # dataset3 = load_dataset('covost2','de_en',data_dir = 'temp_datasets/covost-de_en')['train'].select(range(7000))
        # dataset4 = load_dataset('covost2','fr_en',data_dir = 'temp_datasets/covost-fr_en')['train'].select(range(7000))
        # dataset5 = load_dataset('covost2','mn_en',data_dir = 'temp_datasets/covost-mn_en')['train']
        # dataset6 = load_dataset('covost2','ar_en',data_dir = 'temp_datasets/covost-ar_en')['train']
        # dataset = dataset['train']
        
        # arr = ['dutch','french','german','italian','polish','portuguese','spanish']
        # con_dataset = None
        # for i in arr:
        #     dataset1 = load_dataset("facebook/multilingual_librispeech", i, split="9_hours")
        #     dataset2 = load_dataset("facebook/multilingual_librispeech", i, split="dev")
        #     if(con_dataset == None):
        #         con_dataset = concatenate_datasets([dataset1, dataset2])
        #     else:
        #         con_dataset = concatenate_datasets([con_dataset, dataset1, dataset2])
        # con_dataset = con_dataset.shuffle()
        # dataset = load_dataset("HuggingFaceH4/ultrachat_200k",split="train_sft")#207865
        # dataset = load_from_disk("temp_datasets/ultrachat_speech").shuffle()#55464
        
        # dataset = load_from_disk("temp_datasets/VoiceAssistant").shuffle()  #459067
        # dataset = load_from_disk("temp_datasets/ZHEN_mixed_filtered").shuffle()  #246866
        # dataset = load_from_disk("temp_datasets/chinese_speech").shuffle() #123433
        dataset = load_from_disk("temp_datasets/chinese_speech_only_cosy")
        dataset2 = load_from_disk("temp_datasets/chinese_speech_only_cosy2")
        dataset3 = load_from_disk("temp_datasets/chinese_speech_only_cosy3")
        dataset4 = load_from_disk("temp_datasets/chinese_speech_only_cosy4")
        dataset5 = load_from_disk("temp_datasets/chinese_speech_only_cosy5")
        dataset = concatenate_datasets([dataset, dataset2,dataset3,dataset4,dataset5]).shuffle()#49999
        
        # dataset = load_dataset("carlot/AIShell",split="train")
        # dataset2 = load_dataset("carlot/AIShell",split="validation")
        # dataset = concatenate_datasets([dataset, dataset2])#134424
        
        
        # dataset = load_from_disk("temp_datasets/ZHEN_mixed_filteredd").shuffle()  #246866
        # dataset = load_dataset("JerryAGENDD/ultrachat_tensor_10k", cache_dir="temp_datasets").shuffle()
        # dataset = load_from_disk("temp_datasets/ultrachat_tensor_10000").shuffle()
        # print(len(con_dataset))#29060
        # dataset, transcipt = aishell() # 120098
        
        dataset = MyDataset(args, dataset)
        data_loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=args.micro_bsz, num_workers=8, persistent_workers=False, drop_last=True, collate_fn=lambda x: x)
        print("train starting...")
        # with torch.cuda.amp.autocast():
        trainer.fit(Total_model, data_loader)
        
    elif(args.OP == 2):#自回归
        import soundfile as sf
        # con_dataset = load_from_disk("temp_datasets/VoiceAssistant")
        
        # con_dataset = load_from_disk("temp_datasets/ZHEN_mixed_filtered").shuffle()
        #con_dataset = load_from_disk("temp_datasets/chinese_speech").shuffle()
        con_dataset = load_from_disk("temp_datasets/chinese_speech_only_cosy").shuffle()
        # con_dataset = load_dataset("carlot/AIShell",split="test").shuffle()
        # con_dataset, transcipt = aishell('test')
        
        
        tokenizer = Total_model.return_tokenizer()
        # Total_model = Total_model.to("cuda", dtype=torch.bfloat16)
        Total_model = Total_model.to("cuda", dtype=torch.bfloat16)
        Total_model.eval()
        print("start prediction...")
        count = 0
        
        # with open("temp_audios/text.txt",'w') as f:
        for data in con_dataset:
            # pattern = re.compile(r'[a-zA-Z+=-]')
            # if(pattern.search(data['transcript'])):
            #     continue

                #cosy
            inputs = data['speech_cosy'][0]
            answer = data['answer']
            
                #aishell
            # inputs = data['audio']['array']
            # answer = data['transcription']
            # answer = answer.replace(" ","")
            
            print(f"questions:\n{data['transcript']}")
            print(f"true answer:\n{answer[:100]}...")
            print()
            print("predict:")
            output= Total_model.generate(audios = inputs, dy = True, endding = '<s>',length=100)
            output = "".join(output)
            # sf.write(f'temp_audios/output{count}.wav', inputs, 16000)
            # f.write(f"questions {count}:\n{data['transcript']}\n")
            # f.write(f"true answer:\n{answer[:200]}\n")
            # f.write(f"predict:\n{output[:100]}\n")
            # f.write(f"\n\n")
            # output = ''.join(output)
            print("\n\n")
            count+=1
    elif(args.OP == 3):
        from datasets import load_from_disk
        dataset = load_from_disk("temp_datasets/en-final").select(range(100))
        tokenizer = Total_model.return_tokenizer()
        Total_model.to("cuda", dtype=torch.bfloat16)
        for data in dataset:
            text = data['text'].lower().replace("salary", "apple")
            output,_,_ = Total_model([data['speech']], [text])
            # print(output.shape)
            output_ids = torch.argmax(output, dim=-1)
            output_ids = output_ids.flatten().tolist()
            output = tokenizer.decode(output_ids)
            
            print(f"output:{output}")
            print(f"answer:{data['text'].lower()}")
            print()
            exit(0)
    elif(args.OP == 4):
        from datasets import load_dataset

        arr = ['dutch','french','german','italian','polish','portuguese','spanish']
        con_dataset = None
        for i in arr:
            dataset1 = load_dataset("facebook/multilingual_librispeech", i, split="test").select(range(10))
            if(con_dataset == None):
                con_dataset = dataset1
            else:
                con_dataset = concatenate_datasets([con_dataset, dataset1])
        
        tokenizer = Total_model.return_tokenizer()
        Total_model.to("cuda", dtype=torch.bfloat16)
        dss = [con_dataset]
        
        from jiwer import wer
        def calculate_wer(predictions, references):
            total_wer = 0.0
            for pred, ref in zip(predictions, references):
                total_wer += wer(ref, pred)
            average_wer = total_wer / len(predictions)
            return average_wer
        
        from tqdm import tqdm
        for ds in dss:
            predictions = []
            references = []
            for i in tqdm(range(len(ds))):
                x = ds[i]["audio"]["array"]
                z = ds[i]["transcript"].lower()
                # asr(x)
                # print(f"Audio length:{len(x)/16000} s")
                with torch.no_grad():
                    output = Total_model.generate(x) 
                    # output = Total_model.generate(resampy.resample(x, 48000, 16000))
                output = ''.join(output)
                predictions.append(output)
                references.append(z)
                tqdm.write(f"{output}\t{z}")
            average_wer = calculate_wer(predictions, references)
            # print(ds)
            print(f"Average WER: {average_wer}")
    elif(args.OP == 5):
        from jiwer import cer
        from datasets import load_dataset
        from tqdm import tqdm
        
        # dataset = load_dataset("mozilla-foundation/common_voice_13_0", "zh-CN", split="test",token = token)
        
        dataset, transcipt = aishell(split="test")
        
        # dataset = dataset[:1000]
        
        
        tokenizer = Total_model.return_tokenizer()
        Total_model = Total_model.to("cuda", dtype=torch.bfloat16)
        
        dss = [dataset]
        def calculate_cer(predictions, references):
            total_cer = 0.0
            for pred, ref in zip(predictions, references):
                total_cer += cer(ref, pred)
            average_cer = total_cer / len(predictions)
            return average_cer

        for ds in dss:
            predictions = []
            references = []
            with open("cer_log.txt","w") as f:
                for i in tqdm(range(len(ds))):
                    # x = ds[i]["audio"]["array"]
                    # z = ds[i]["sentence"].lower()
                    # # asr(x)
                    # # print(f"Audio length:{len(x)/16000} s")
                    
                    path = 'temp_datasets/aishell/data_aishell/wav/test/'
                    sr, audio = wav.read(path+ds[i]+".wav")
                    x = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=16000)
                    z = transcipt[ds[i]].replace(" ","")
                    with torch.no_grad():
                        output = Total_model.generate(x) 
                        # output = Total_model.generate(resampy.resample(x, 48000, 16000))
                    
                    output = ''.join(output)
                    output = output.replace(" ","")
                    predictions.append(output)
                    references.append(z)
                    tqdm.write(f"{output}\t{z}")
                    f.write(f"{output}\t{z}\n")
            
            average_cer = calculate_cer(predictions, references)
            # print(ds)
            print(f"Average CER: {average_cer}")
    elif(args.OP == 6):
        
        import librosa
        import time
        
        audio, sr = librosa.load("output.wav", sr=None)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        Total_model = Total_model.to("cuda", dtype=torch.bfloat16)
        
        start_time = time.time()
        output= Total_model.generate(audio)
        output = ''.join(output)
        end_time = time.time()
        
        # print(f"audio: {args.file_path}")
        print(f"predict: {output}")
        print(f"Response time: {end_time - start_time} seconds")
    exit(0)

