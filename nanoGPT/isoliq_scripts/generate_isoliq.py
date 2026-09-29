import argparse
from contextlib import nullcontext

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--tokenizer', default='/content/drive/MyDrive/Isoliq Inno 1/artifacts/tokenizer.json')
parser.add_argument('--prompt', required=True)
parser.add_argument('--chat', action='store_true')
parser.add_argument('--system', default='You are Isoliq Inno 1, a small helpful AI assistant.')
parser.add_argument('--max_new_tokens', type=int, default=120)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--seed', type=int, default=1337)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False,
)

config = GPTConfig(**checkpoint['model_args'])
model = GPT(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for key in list(state_dict):
    if key.startswith(unwanted_prefix):
        state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

tokenizer = Tokenizer.from_file(args.tokenizer)
if args.chat:
    prompt = (
        f'<|im_start|>system\n{args.system}<|im_end|>\n'
        f'<|im_start|>user\n{args.prompt}<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )
else:
    prompt = args.prompt

prompt_ids = tokenizer.encode(prompt).ids[-config.block_size:]
x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dtype = (
    torch.bfloat16
    if device == 'cuda' and torch.cuda.is_bf16_supported()
    else torch.float16
)
context = torch.amp.autocast('cuda', dtype=dtype) if device == 'cuda' else nullcontext()
with torch.no_grad(), context:
    y = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

generated_ids = y[0, x.shape[1]:].tolist()
answer = tokenizer.decode(generated_ids, skip_special_tokens=False)
answer = answer.split('<|im_end|>', 1)[0]
answer = answer.split('<|endoftext|>', 1)[0]
print(answer.strip() or '(the model ended immediately)')
