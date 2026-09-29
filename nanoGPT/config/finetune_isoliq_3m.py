out_dir = '/content/drive/MyDrive/Isoliq Inno 1/checkpoints/finetune'

log_interval = 5
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'resume'

wandb_log = False

dataset = 'isoliq_sft'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 512

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.1
bias = False

learning_rate = 5e-5
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

train_tokens = os.path.getsize('data/isoliq_sft/train.bin') // 2
tokens_per_iteration = gradient_accumulation_steps * batch_size * block_size
max_iters = max(50, math.ceil(2 * train_tokens / tokens_per_iteration))
eval_interval = max(20, max_iters // 5)

decay_lr = True
warmup_iters = max(5, max_iters // 20)
lr_decay_iters = max_iters
min_lr = 5e-6

device = 'cuda'
compile = False

print('Fine-tuning train tokens:', train_tokens)
print('Fine-tuning max_iters:', max_iters)
