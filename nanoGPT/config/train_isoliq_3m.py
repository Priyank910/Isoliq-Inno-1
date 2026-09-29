out_dir = '/content/drive/MyDrive/Isoliq Inno 1/checkpoints/pretrain'

eval_interval = 100
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

wandb_log = False

dataset = 'isoliq'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 1000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 50
lr_decay_iters = 1000
min_lr = 6e-5

device = 'cuda'
compile = False
