load_model='temp_models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
proj_dir='output'
data_file='temp_datasets'


n_layer=24
n_embd=2048

micro_bsz=16
epoch_save=1
epoch_steps=17500
ctx_len=1024

QUANT='nf4' 

python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--data_type binidx --vocab_size 65536 \
--ctx_len $ctx_len --epoch_steps $epoch_steps --epoch_count 10 --epoch_begin 0 --epoch_save $epoch_save --micro_bsz $micro_bsz \
--n_layer $n_layer --n_embd $n_embd \
--pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-3 --warmup_steps 100 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --strategy deepspeed_stage_1 \
--precision bf16 \
--my_testing "x060" \
--train_type "state"  --dataload pad --quant $QUANT \
--grad_cp 1 \