from pathlib import Path
import torch

from model import GPT, GPTConfig

ROOT = Path('/content/drive/MyDrive/Isoliq Inno 1')
PRETRAINED = ROOT / 'checkpoints/pretrain/ckpt.pt'
OUT_DIR = ROOT / 'checkpoints/finetune'
OUT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = torch.load(PRETRAINED, map_location='cpu', weights_only=False)
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for key in list(state_dict):
    if key.startswith(unwanted_prefix):
        state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
model.load_state_dict(state_dict)

optimizer = model.configure_optimizers(
    weight_decay=0.01,
    learning_rate=5e-5,
    betas=(0.9, 0.95),
    device_type='cpu',
)

seed = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': 0,
    'best_val_loss': 1e9,
    'config': {},
}
torch.save(seed, OUT_DIR / 'ckpt.pt')
print('Created:', OUT_DIR / 'ckpt.pt')
