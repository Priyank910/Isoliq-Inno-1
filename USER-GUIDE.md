# Isoliq Inno 1: Google Colab User Guide

This guide reproduces the **Isoliq Inno 1** experiment shown in the video: prepare a text corpus, train a compact nanoGPT model from scratch, fine-tune it for simple chat, and test what it actually learned.

It is written for curious builders and beginner machine-learning practitioners. You do not need a local GPU, but you should be comfortable running notebook cells and reading error messages.

> **Experiment, not a production model:** Isoliq Inno 1 has only **3,000,768 trainable parameters**. It is useful for learning and inspection, but it is not a replacement for a modern assistant and should not be trusted for factual or safety-critical answers.

## What you will build

| Stage | What happens | Main output |
| --- | --- | --- |
| Setup | Mount Drive, check the GPU, install dependencies, and pin nanoGPT | Persistent project workspace |
| Pretraining data | Clean the source corpus, train a tokenizer, and encode train/validation splits | `train.bin`, `val.bin`, `tokenizer.json` |
| Pretraining | Train the 3M-parameter base model as a text completer | `checkpoints/pretrain/ckpt.pt` |
| Fine-tuning data | Remove duplicates and conflicting prompt groups, then create assistant-only loss masks | `data/isoliq_sft/*` and an audit report |
| Fine-tuning | Start from the pretrained weights with fresh optimizer state | `checkpoints/finetune/ckpt.pt` |
| Evaluation | Test several prompts and random seeds instead of trusting one sample | Comparable model outputs |

## Before you begin

You need:

- a Google account with access to Google Drive and Google Colab;
- a Colab runtime with a GPU enabled under **Runtime > Change runtime type > Hardware accelerator > GPU**;
- `input.txt`, the corpus used for pretraining;
- `finetune_data.txt`, the original ChatML conversation file;
- `finetune_data2.txt`, the prepared conflict-filtered copy used later in this guide;
- enough Drive space for the datasets, encoded binaries, scripts, and checkpoints.

Upload the first two source files before starting:

```text
/content/drive/MyDrive/Isoliq Inno 1/data/input.txt
/content/drive/MyDrive/Isoliq Inno 1/data/finetune_data.txt
```

Step 13 explains when and where to upload `finetune_data2.txt`.

## How to use this guide

1. Create a new Google Colab notebook.
2. Enable a GPU runtime.
3. Run each numbered section in order the first time.
4. Treat every separate code fence as its own Colab cell unless the text says otherwise.
5. Stop when an assertion fails or an expected file is missing. Fix that step before continuing.
6. After a runtime disconnect, use the recovery commands near the end instead of repeating the entire setup.

### Colab notation used below

| Prefix | Meaning |
| --- | --- |
| `!command` | Run a Linux shell command from a notebook cell |
| `%cd path` | Change the notebook's current working directory |
| `%%writefile path` | Save the rest of that cell as a file |
| Plain Python | Run directly in the active Python notebook process |

## Complete workflow

Run these cells in order in a new Google Colab notebook. All source files, scripts, tokenized data, configurations, and checkpoints stay inside:

```text
/content/drive/MyDrive/Isoliq Inno 1
```

The only non-persistent part is the Colab Python environment, so the package-install cell must be run again after a runtime restart.

The selected architecture uses a 1,280-token vocabulary and has **3,000,768 trainable parameters**:

```text
n_layer = 6
n_head = 6
n_embd = 192
block_size = 512
dropout = 0.0
bias = False
vocab_size = 1280
```

### Run order at a glance

| Step | Action | Why it matters |
| ---: | --- | --- |
| 1 | Mount Google Drive | Keep data and checkpoints after the runtime ends |
| 2 | Create project folders | Give every script one stable path |
| 3 | Check GPU and install packages | Verify the runtime before expensive work |
| 4 | Clone and pin nanoGPT | Use the same source revision as the experiment |
| 5 | Fingerprint the data | Record exactly which files are being used |
| 6 | Prepare pretraining data | Train the tokenizer and encode the corpus |
| 7 | Verify parameters | Confirm the model is exactly the documented size |
| 8 | Patch nanoGPT | Support current checkpoint loading and assistant masks |
| 9 | Write pretraining config | Save reproducible model and optimizer settings |
| 10 | Run pretraining smoke test | Catch setup errors cheaply |
| 11 | Run full pretraining | Produce the base text-completion checkpoint |
| 12 | Create generation utility | Test checkpoints with the correct tokenizer |
| 13 | Prepare fine-tuning data | Remove known conflicts and create loss masks |
| 14 | Create fresh SFT seed | Keep weights while resetting training state |
| 15 | Write fine-tuning config | Calculate a data-relative run length |
| 16 | Run fine-tuning smoke test | Verify masks and checkpoint loading |
| 17 | Run full fine-tuning | Produce the chat-format checkpoint |
| 18 | Evaluate responses | Compare prompts and seeds honestly |

## 1. Mount Google Drive

Connect Colab to persistent storage. Approve the Google authorization prompt when it appears.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Define the persistent project folder

Create the folder structure used by every later command and expose its path as the `PROJECT` environment variable.

```python
import os
from pathlib import Path

PROJECT = Path('/content/drive/MyDrive/Isoliq Inno 1')
os.environ['PROJECT'] = str(PROJECT)

for folder in [
    PROJECT / 'data',
    PROJECT / 'artifacts',
    PROJECT / 'checkpoints' / 'pretrain',
    PROJECT / 'checkpoints' / 'finetune',
]:
    folder.mkdir(parents=True, exist_ok=True)

print('Project:', PROJECT)
print('input.txt:', (PROJECT / 'data/input.txt').exists())
print('finetune_data.txt:', (PROJECT / 'data/finetune_data.txt').exists())
```

Both final lines must print `True`.

## 3. Inspect the GPU and install packages

Do not reinstall PyTorch. Colab already supplies a CUDA-compatible build.

The first cell confirms that Colab can see the GPU and reports its memory and BF16 support. The second installs the data, tokenizer, and experiment utilities used by the scripts.

```python
!nvidia-smi

import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else 0)
print('BF16 supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
```

```python
!pip install -q numpy transformers datasets tiktoken wandb tqdm tokenizers==0.23.2
```

## 4. Clone nanoGPT directly into the persistent folder

Run this once. The `test` command prevents a second clone if the folder already exists.

```python
!test -d "$PROJECT/nanoGPT/.git" || git clone https://github.com/karpathy/nanoGPT.git "$PROJECT/nanoGPT"
!git -C "$PROJECT/nanoGPT" checkout 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
```

```python
%cd "/content/drive/MyDrive/Isoliq Inno 1/nanoGPT"
!mkdir -p isoliq_scripts data/isoliq data/isoliq_sft
!touch isoliq_scripts/__init__.py
!git rev-parse HEAD
```

The final command should print:

```text
3adf61e154c3fe3fca428ad6bc3818b27a3b8291
```

## 5. Check the uploaded data

Confirm both source files exist, record their sizes, and calculate SHA-256 fingerprints so you know exactly which data produced a run.

```python
!ls -lh "$PROJECT/data/input.txt" "$PROJECT/data/finetune_data.txt"
!sha256sum "$PROJECT/data/input.txt" "$PROJECT/data/finetune_data.txt"
```

Expected hashes for the files inspected while this guide was created:

```text
input.txt          46ba0c36d81845bc1517f5de1653515ac3a60d3ed98603dd982b72a31606e294
finetune_data.txt  dac3dd60f20c0c0bc3db5251b2715a35c6548c5d74a80a4af8de8047102752fa
```

If your hashes differ, the commands can still run, but you are using different data.

## 6. Create the tokenizer and pretraining binaries

This creates a 1,280-token byte-level BPE tokenizer, a deterministic 95/5 document split, `train.bin`, `val.bin`, and `meta.pkl`.

```python
%%writefile isoliq_scripts/prepare_pretrain.py
from pathlib import Path
import hashlib
import json
import pickle
import random
import unicodedata

import numpy as np
from tokenizers import AddedToken, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

ROOT = Path('/content/drive/MyDrive/Isoliq Inno 1')
SOURCE = ROOT / 'data/input.txt'
OUT = ROOT / 'nanoGPT/data/isoliq'
ARTIFACTS = ROOT / 'artifacts'

SEED = 1337
VOCAB_SIZE = 1280
VAL_FRACTION = 0.05
EOT = '<|endoftext|>'
SPECIAL_TOKENS = [EOT, '<|im_start|>', '<|im_end|>']

OUT.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

raw = SOURCE.read_bytes()
text = raw.decode('utf-8')
text = text.replace('\r\n', '\n').replace('\r', '\n')
text = unicodedata.normalize('NFC', text)
replacement_count = text.count('\ufffd')
text = text.replace('\ufffd', ' ')

parts = text.split(EOT)
if len(parts) < 2:
    raise ValueError(f'No {EOT} document separators were found')

# Ignore anything after the last complete separator. The inspected source ends
# with an incomplete article fragment.
documents = [part.strip() for part in parts[:-1] if part.strip()]
if len(documents) < 2:
    raise ValueError('At least two complete documents are required')

rng = random.Random(SEED)
rng.shuffle(documents)
val_count = max(1, round(len(documents) * VAL_FRACTION))
val_documents = documents[:val_count]
train_documents = documents[val_count:]

specials = [
    AddedToken(token, special=True, normalized=False)
    for token in SPECIAL_TOKENS
]

tokenizer = Tokenizer(BPE(unk_token=None))
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    show_progress=True,
    special_tokens=specials,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tokenizer.train_from_iterator(
    train_documents,
    trainer=trainer,
    length=len(train_documents),
)

actual_vocab_size = tokenizer.get_vocab_size()
if actual_vocab_size != VOCAB_SIZE:
    raise ValueError(f'Expected vocab {VOCAB_SIZE}, got {actual_vocab_size}')

eot_id = tokenizer.token_to_id(EOT)
if eot_id is None:
    raise ValueError('EOT token is missing from the tokenizer')

tokenizer.save(str(ARTIFACTS / 'tokenizer.json'))
tokenizer.save(str(OUT / 'tokenizer.json'))

def write_split(name, docs, batch_documents=32):
    destination = OUT / f'{name}.bin'
    token_count = 0
    with destination.open('wb') as output:
        for start in range(0, len(docs), batch_documents):
            token_ids = []
            for document in docs[start:start + batch_documents]:
                token_ids.extend(tokenizer.encode(document).ids)
                token_ids.append(eot_id)
            ids = np.asarray(token_ids, dtype=np.uint16)
            ids.tofile(output)
            token_count += len(ids)
    return token_count

train_tokens = write_split('train', train_documents)
val_tokens = write_split('val', val_documents)

meta = {
    'vocab_size': actual_vocab_size,
    'dtype': 'uint16',
    'tokenizer': 'tokenizer.json',
    'special_tokens': {
        token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
    },
}
with (OUT / 'meta.pkl').open('wb') as file:
    pickle.dump(meta, file)

audit = {
    'source': str(SOURCE),
    'source_sha256': hashlib.sha256(raw).hexdigest(),
    'documents': len(documents),
    'train_documents': len(train_documents),
    'val_documents': len(val_documents),
    'train_tokens': train_tokens,
    'val_tokens': val_tokens,
    'vocab_size': actual_vocab_size,
    'replacement_characters_removed': replacement_count,
    'seed': SEED,
}
(ARTIFACTS / 'pretrain_audit.json').write_text(
    json.dumps(audit, indent=2), encoding='utf-8'
)
print(json.dumps(audit, indent=2))
```

Run it:

```python
!python -m isoliq_scripts.prepare_pretrain
!ls -lh data/isoliq "$PROJECT/artifacts/tokenizer.json"
```

This preprocessing cell may take several minutes. Do not interrupt it while Drive files are being written.

## 7. Verify the exact parameter count

Instantiate the configured network without training it, then count every trainable value. The assertions stop the workflow if the architecture drifts from the documented 3M model.

```python
%%writefile isoliq_scripts/verify_parameters.py
from model import GPT, GPTConfig

config = GPTConfig(
    block_size=512,
    vocab_size=1280,
    n_layer=6,
    n_head=6,
    n_embd=192,
    dropout=0.0,
    bias=False,
)

model = GPT(config)
all_parameters = sum(parameter.numel() for parameter in model.parameters())
nanoGPT_reported = model.get_num_params()

print(f'All trainable parameters: {all_parameters:,}')
print(f'nanoGPT default report:  {nanoGPT_reported:,}')
print('The nanoGPT default excludes learned position embeddings.')

assert all_parameters == 3_000_768
assert nanoGPT_reported == 2_902_464
```

```python
!python -m isoliq_scripts.verify_parameters
```

## 8. Patch nanoGPT for current PyTorch and assistant-only loss

The patch is required for trusted checkpoints on PyTorch 2.6 or newer and for applying loss only to assistant tokens during fine-tuning. It leaves ordinary pretraining unchanged.

```python
%%writefile isoliq_scripts/patch_nanogpt.py
from pathlib import Path
import shutil

path = Path('train.py')
backup = Path('train.py.isoliq_original')
source = path.read_text(encoding='utf-8')

if not backup.exists():
    shutil.copy2(path, backup)

old_load = "checkpoint = torch.load(ckpt_path, map_location=device)"
new_load = "checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)"
if old_load in source:
    source = source.replace(old_load, new_load)

if "f'{split}_mask.bin'" not in source:
    start = source.index('def get_batch(split):')
    end = source.index('\n# init these up here', start)
    replacement = '''def get_batch(split):
    data_path = os.path.join(data_dir, f'{split}.bin')
    mask_path = os.path.join(data_dir, f'{split}_mask.bin')

    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    loss_mask = (
        np.memmap(mask_path, dtype=np.uint8, mode='r')
        if os.path.exists(mask_path)
        else None
    )

    if len(data) <= block_size:
        raise ValueError(f'{split}.bin must contain more than block_size tokens')
    if loss_mask is not None and len(loss_mask) != len(data):
        raise ValueError(f'{split} data and mask lengths differ')

    if loss_mask is None:
        starts = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    else:
        starts = []
        attempts = 0
        max_attempts = batch_size * 1000
        while len(starts) < batch_size:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(f'Could not find supervised targets in {split}_mask.bin')
            candidate = int(torch.randint(len(data) - block_size, (1,)).item())
            if loss_mask[candidate + 1:candidate + 1 + block_size].any():
                starts.append(candidate)

    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64))
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
        for i in starts
    ])

    if loss_mask is not None:
        target_mask = torch.stack([
            torch.from_numpy(
                loss_mask[i + 1:i + 1 + block_size].astype(np.bool_)
            )
            for i in starts
        ])
        y[~target_mask] = -1

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
'''
    source = source[:start] + replacement + source[end:]

path.write_text(source, encoding='utf-8')
print('nanoGPT patch is ready')
```

```python
!python -m isoliq_scripts.patch_nanogpt
!python -m py_compile train.py
```

Run the patch command again after intentionally restoring or updating `train.py`.

## 9. Create the 3M pretraining configuration

Write nanoGPT's model, batch, optimizer, evaluation, checkpoint, and learning-rate settings into a reusable configuration file.

```python
%%writefile config/train_isoliq_3m.py
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
```

The effective batch is `32 × 4 × 512 = 65,536` tokens per optimizer update. This should fit comfortably below 15 GB for this model.

## 10. Run a short pretraining smoke test

Run this before the full training job:

```python
%cd "/content/drive/MyDrive/Isoliq Inno 1/nanoGPT"
!python train.py config/train_isoliq_3m.py --out_dir="$PROJECT/checkpoints/pretrain-smoke" --max_iters=20 --eval_interval=10 --eval_iters=10
```

The output should show finite training and validation losses and no CUDA or token-index errors.

This is intentionally a short smoke test, not the real training run. Because the
configuration has `log_interval = 10`, nanoGPT prints progress at iterations 0,
10, and 20 instead of printing every update. With nanoGPT's inclusive loop this
performs update indexes 0 through 20, then stops. Continue to Step 11 for the
1,000-iteration training run.

## 11. Start full pretraining

```python
%cd "/content/drive/MyDrive/Isoliq Inno 1/nanoGPT"
!python train.py config/train_isoliq_3m.py
```

The checkpoint is stored persistently at:

```text
/content/drive/MyDrive/Isoliq Inno 1/checkpoints/pretrain/ckpt.pt
```

### If CUDA runs out of memory

Try the same effective batch with a smaller micro-batch:

```python
!python train.py config/train_isoliq_3m.py --batch_size=16 --gradient_accumulation_steps=8
```

If necessary:

```python
!python train.py config/train_isoliq_3m.py --batch_size=8 --gradient_accumulation_steps=16
```

### Resume after a Colab disconnection

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!pip install -q numpy transformers datasets tiktoken wandb tqdm tokenizers==0.23.2
%cd "/content/drive/MyDrive/Isoliq Inno 1/nanoGPT"
!python -m isoliq_scripts.patch_nanogpt
!python train.py config/train_isoliq_3m.py --init_from=resume
```

`max_iters` is the final iteration number, not the number of extra iterations. To extend a completed 1,000-iteration run to 1,500:

```python
!python train.py config/train_isoliq_3m.py --init_from=resume --max_iters=1500 --lr_decay_iters=1500
```

## 12. Create a tokenizer-aware generation script

```python
%%writefile isoliq_scripts/generate_isoliq.py
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
```

Test the pretrained model as a text completer:

```python
!python -m isoliq_scripts.generate_isoliq --checkpoint "$PROJECT/checkpoints/pretrain/ckpt.pt" --prompt "The history of computing"
```

Do not expect chat behavior yet. Pretraining teaches text continuation; fine-tuning teaches the ChatML conversation format.

## 13. Clean and prepare the fine-tuning data

The inspected `finetune_data.txt` contains 9,116 parseable conversations, but most are duplicate timeline prompts and 6,819 examples are inside prompt groups with contradictory answers. It also contains no user greeting prompt such as `hello`.

The following preparation script:

- removes exact duplicates;
- removes every normalized user prompt that maps to conflicting answers;
- adds a small basic-chat set, including greetings;
- creates assistant-only training masks;
- reuses the exact pretraining tokenizer.

### Use the prepared finetune_data2 copy

The original `finetune_data.txt` is preserved. A separate cleaned copy was
created locally at:

```text
D:\Projects\InnoDev\Training-Data\finetune_data2.txt
```

Upload that file to this exact Drive location before continuing:

```text
/content/drive/MyDrive/Isoliq Inno 1/data/finetune_data2.txt
```

Verify the upload in Colab:

```python
!ls -lh "$PROJECT/data/finetune_data.txt" "$PROJECT/data/finetune_data2.txt"
!sha256sum "$PROJECT/data/finetune_data2.txt"
```

Expected SHA-256 for the prepared copy:

```text
27d174363fcb67e88635e83e7a39f0bacf9c1e4ac5f10d234211cb556a8a883b
```

The prepared copy contains 2,308 conversations: 2,292 safe conversations
retained from the original data plus 16 basic-chat conversations. It has zero
exact duplicate conversations, zero conflicting normalized user prompts, and
zero empty ChatML fields. The text file cannot contain loss masks itself; the
existing preparation script below creates `train_mask.bin` and `val_mask.bin`
with the original pretraining tokenizer. The script automatically uses
`finetune_data2.txt` when it exists and falls back to the original file only
when the prepared copy has not been uploaded.

```python
%%writefile isoliq_scripts/prepare_finetune.py
from collections import defaultdict
from pathlib import Path
import json
import pickle
import random
import re
import shutil

import numpy as np
from tokenizers import Tokenizer

ROOT = Path('/content/drive/MyDrive/Isoliq Inno 1')
SOURCE = ROOT / 'data/finetune_data.txt'
PREPARED_SOURCE = ROOT / 'data/finetune_data2.txt'
if PREPARED_SOURCE.exists():
    SOURCE = PREPARED_SOURCE
CLEAN_SOURCE = ROOT / 'data/finetune_data_clean.txt'
TOKENIZER_PATH = ROOT / 'artifacts/tokenizer.json'
PRETRAIN_META = ROOT / 'nanoGPT/data/isoliq/meta.pkl'
OUT = ROOT / 'nanoGPT/data/isoliq_sft'
ARTIFACTS = ROOT / 'artifacts'

SEED = 1337
VAL_FRACTION = 0.10
EOT = '<|endoftext|>'

OUT.mkdir(parents=True, exist_ok=True)
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

text = SOURCE.read_text(encoding='utf-8')
text = text.replace('\r\n', '\n').replace('\r', '\n')
pattern = re.compile(
    r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n'
    r'<\|im_start\|>user\n(.*?)<\|im_end\|>\n'
    r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
    re.S,
)
records = [tuple(value.strip() for value in match.groups()) for match in pattern.finditer(text)]
if not records:
    raise ValueError('No complete ChatML conversations were found')
if any(not system or not user or not assistant for system, user, assistant in records):
    raise ValueError('Every system, user, and assistant field must be non-empty')

unique_records = list(dict.fromkeys(records))

def normalize_user(value):
    return ' '.join(value.casefold().split())

by_user = defaultdict(list)
for record in unique_records:
    by_user[normalize_user(record[1])].append(record)

conflicting_questions = {
    question
    for question, group in by_user.items()
    if len({record[2] for record in group}) > 1
}
conflicting_examples = sum(
    1 for record in records if normalize_user(record[1]) in conflicting_questions
)

clean_records = [
    group[0]
    for question, group in by_user.items()
    if question not in conflicting_questions
]

system_prompt = 'You are Isoliq Inno 1, a small helpful AI assistant.'
basics = [
    (system_prompt, 'Hello', 'Hello! How can I help you?'),
    (system_prompt, 'Hi', 'Hi! What would you like to talk about?'),
    (system_prompt, 'Hey', 'Hey! How can I help?'),
    (system_prompt, 'Good morning', 'Good morning! How can I help you today?'),
    (system_prompt, 'Good evening', 'Good evening! What can I help you with?'),
    (system_prompt, 'How are you?', "I'm ready to help. How are you?"),
    (system_prompt, 'What is your name?', "I'm Isoliq Inno 1, a small language model."),
    (system_prompt, 'Who are you?', "I'm Isoliq Inno 1, a small AI assistant."),
    (system_prompt, 'What can you do?', 'I can answer simple questions and discuss topics covered by my training data.'),
    (system_prompt, 'Thank you', "You're welcome!"),
    (system_prompt, 'Thanks', "You're welcome!"),
    (system_prompt, 'Goodbye', 'Goodbye!'),
    (system_prompt, 'Bye', 'Bye! Take care.'),
    (system_prompt, 'Can you help me?', 'Yes. Tell me what you need help with.'),
    (system_prompt, 'Are you a human?', "No. I'm a small language model called Isoliq Inno 1."),
    (system_prompt, 'Tell me something about yourself', "I'm Isoliq Inno 1, a compact experimental language model."),
]

rng = random.Random(SEED)
rng.shuffle(clean_records)
val_count = max(1, round(len(clean_records) * VAL_FRACTION))
val_records = clean_records[:val_count]
train_records = clean_records[val_count:] + basics * 5
rng.shuffle(train_records)

def serialize(record):
    system, user, assistant = record
    return (
        f'<|im_start|>system\n{system}<|im_end|>\n'
        f'<|im_start|>user\n{user}<|im_end|>\n'
        f'<|im_start|>assistant\n{assistant}<|im_end|>\n'
    )

CLEAN_SOURCE.write_text(
    '\n'.join(serialize(record) for record in clean_records + basics),
    encoding='utf-8',
)

def encode_record(record):
    system, user, assistant = record
    prefix = (
        f'<|im_start|>system\n{system}<|im_end|>\n'
        f'<|im_start|>user\n{user}<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )
    suffix = f'{assistant}<|im_end|>{EOT}'
    prefix_ids = tokenizer.encode(prefix).ids
    suffix_ids = tokenizer.encode(suffix).ids
    ids = prefix_ids + suffix_ids
    mask = [0] * len(prefix_ids) + [1] * len(suffix_ids)
    return ids, mask

def write_split(name, split_records):
    token_count = 0
    supervised_count = 0
    with (OUT / f'{name}.bin').open('wb') as data_file, \
         (OUT / f'{name}_mask.bin').open('wb') as mask_file:
        for record in split_records:
            ids, mask = encode_record(record)
            np.asarray(ids, dtype=np.uint16).tofile(data_file)
            np.asarray(mask, dtype=np.uint8).tofile(mask_file)
            token_count += len(ids)
            supervised_count += sum(mask)
    return token_count, supervised_count

train_tokens, train_supervised = write_split('train', train_records)
val_tokens, val_supervised = write_split('val', val_records)

shutil.copy2(PRETRAIN_META, OUT / 'meta.pkl')
shutil.copy2(TOKENIZER_PATH, OUT / 'tokenizer.json')

report = {
    'parsed_conversations': len(records),
    'exact_unique_conversations': len(unique_records),
    'conflicting_questions_removed': len(conflicting_questions),
    'examples_inside_conflict_groups': conflicting_examples,
    'clean_source_conversations': len(clean_records),
    'basic_chat_examples': len(basics),
    'basic_chat_training_repetitions': 5,
    'train_conversations': len(train_records),
    'val_conversations': len(val_records),
    'train_tokens': train_tokens,
    'train_supervised_tokens': train_supervised,
    'val_tokens': val_tokens,
    'val_supervised_tokens': val_supervised,
}
(ARTIFACTS / 'finetune_audit.json').write_text(
    json.dumps(report, indent=2), encoding='utf-8'
)
print(json.dumps(report, indent=2))
```

Run it and verify the masks:

```python
!python -m isoliq_scripts.prepare_finetune
!ls -lh data/isoliq_sft "$PROJECT/data/finetune_data_clean.txt" "$PROJECT/artifacts/finetune_audit.json"
```

```python
import numpy as np
from pathlib import Path

sft = Path('/content/drive/MyDrive/Isoliq Inno 1/nanoGPT/data/isoliq_sft')
for split in ['train', 'val']:
    data = np.memmap(sft / f'{split}.bin', dtype=np.uint16, mode='r')
    mask = np.memmap(sft / f'{split}_mask.bin', dtype=np.uint8, mode='r')
    print(split, 'tokens:', len(data), 'supervised:', int(mask.sum()))
    assert len(data) == len(mask)
    assert mask.sum() > 0
    assert set(np.unique(mask)).issubset({0, 1})
```

## 14. Create a fresh fine-tuning checkpoint from the pretrained model

This keeps the pretrained weights but resets the iteration number, validation score, and optimizer state.

```python
%%writefile isoliq_scripts/make_finetune_seed.py
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
```

```python
!python -m isoliq_scripts.make_finetune_seed
!ls -lh "$PROJECT/checkpoints/finetune/ckpt.pt"
```

## 15. Create the fine-tuning configuration

This calculates approximately two passes over the cleaned training tokens instead of guessing a fixed iteration count.

```python
%%writefile config/finetune_isoliq_3m.py
import math
import os

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
```

## 16. Run a fine-tuning smoke test

The smoke test uses a separate checkpoint folder, so it cannot overwrite the real fine-tuning seed.

```python
!mkdir -p "$PROJECT/checkpoints/finetune-smoke"
!cp "$PROJECT/checkpoints/finetune/ckpt.pt" "$PROJECT/checkpoints/finetune-smoke/ckpt.pt"
!python train.py config/finetune_isoliq_3m.py --out_dir="$PROJECT/checkpoints/finetune-smoke" --max_iters=10 --eval_interval=5 --eval_iters=5
```

The output should show finite losses and no `mask length` or `supervised targets` errors.

## 17. Start full fine-tuning

If the smoke test worked, recreate the clean seed once so the real run starts at iteration zero:

```python
!python -m isoliq_scripts.make_finetune_seed
!python train.py config/finetune_isoliq_3m.py
```

The final chat checkpoint remains at:

```text
/content/drive/MyDrive/Isoliq Inno 1/checkpoints/finetune/ckpt.pt
```

Resume an interrupted fine-tuning run with:

```python
!python train.py config/finetune_isoliq_3m.py --init_from=resume
```

Do not run `make_finetune_seed.py` before resuming; that command intentionally resets the run.

## 18. Test whether it can reply to “Hello”

Load the final checkpoint in chat mode and compare multiple prompts and seeds. A single good or bad sample is not enough to judge the model.

```python
!python -m isoliq_scripts.generate_isoliq --chat --checkpoint "$PROJECT/checkpoints/finetune/ckpt.pt" --prompt "Hello" --temperature=0.7 --top_k=30
```

Also test a few variations:

```python
!python -m isoliq_scripts.generate_isoliq --chat --checkpoint "$PROJECT/checkpoints/finetune/ckpt.pt" --prompt "Hi"
!python -m isoliq_scripts.generate_isoliq --chat --checkpoint "$PROJECT/checkpoints/finetune/ckpt.pt" --prompt "What is your name?"
!python -m isoliq_scripts.generate_isoliq --chat --checkpoint "$PROJECT/checkpoints/finetune/ckpt.pt" --prompt "What is anarchism?"
```

Try several seeds instead of judging the model from one sample:

```python
for seed in [1, 2, 3, 4, 5]:
    !python -m isoliq_scripts.generate_isoliq --chat --checkpoint "$PROJECT/checkpoints/finetune/ckpt.pt" --prompt "Hello" --seed={seed} --temperature=0.7 --top_k=30
```

## What to realistically expect

- The pretrained checkpoint is a text completer, not a chatbot.
- The cleaned fine-tuning data plus the added greeting examples should make a simple response to `Hello` likely.
- A 3M model can learn formatting, greetings, short answers, and patterns from a narrow dataset.
- It will still hallucinate, repeat itself, forget context, and fail on many questions.
- If greeting replies are unstable, increase `basic_chat_training_repetitions` from `5` to `10`, rerun Steps 13–18, and compare several fixed prompts.
- Do not fine-tune on the original file unchanged. Its contradictory timeline examples can directly teach the model that the same prompt has many unrelated answers.

## Minimal command order for future Colab sessions

After the first complete setup, a disconnected training session usually needs only:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!pip install -q numpy transformers datasets tiktoken wandb tqdm tokenizers==0.23.2
%cd "/content/drive/MyDrive/Isoliq Inno 1/nanoGPT"
!python -m isoliq_scripts.patch_nanogpt
```

Resume pretraining:

```python
!python train.py config/train_isoliq_3m.py --init_from=resume
```

Or resume fine-tuning:

```python
!python train.py config/finetune_isoliq_3m.py --init_from=resume
```

## Command reference

This table explains the commands viewers will encounter throughout the notebook.

| Command or pattern | What it does | When to use it |
| --- | --- | --- |
| `drive.mount('/content/drive')` | Authorizes Colab to access Google Drive | At the start of every new runtime |
| `Path(...).mkdir(parents=True, exist_ok=True)` | Creates missing project folders without failing when they already exist | Initial setup |
| `os.environ['PROJECT'] = ...` | Makes the project path available to shell commands as `$PROJECT` | Initial setup and path reuse |
| `!nvidia-smi` | Displays the assigned NVIDIA GPU, driver, memory, and active processes | Before training or when diagnosing GPU problems |
| `torch.cuda.is_available()` | Confirms whether PyTorch can use CUDA | Before any training run |
| `torch.cuda.is_bf16_supported()` | Checks whether the GPU supports BF16 mixed precision | Runtime inspection |
| `!pip install -q ...` | Installs tokenizer, dataset, and utility packages quietly | Once per fresh runtime |
| `!test -d ... || git clone ...` | Clones nanoGPT only when its Git folder does not already exist | First setup or after deleting the repository |
| `!git -C ... checkout <commit>` | Pins nanoGPT to the exact revision used by this experiment | Reproducible setup |
| `%cd ".../nanoGPT"` | Changes the notebook's working directory to the nanoGPT repository | Before running scripts or training |
| `!mkdir -p ...` | Creates one or more Linux directories and leaves existing ones intact | Repository setup and smoke-test folders |
| `!touch isoliq_scripts/__init__.py` | Marks `isoliq_scripts` as an importable Python package | Repository setup |
| `!git rev-parse HEAD` | Prints the active Git commit | Verify that the pinned revision is active |
| `!ls -lh ...` | Lists files with human-readable sizes | Confirm uploads and generated artifacts |
| `!sha256sum ...` | Calculates a content fingerprint for a file | Dataset verification and reproducibility |
| `%%writefile path` | Saves the rest of the notebook cell to a file | Create scripts and config files from Colab |
| `!python -m isoliq_scripts.prepare_pretrain` | Cleans, splits, tokenizes, and encodes the pretraining corpus | After writing `prepare_pretrain.py` |
| `!python -m isoliq_scripts.verify_parameters` | Builds the configured model and verifies both parameter counts | Before training |
| `!python -m isoliq_scripts.patch_nanogpt` | Applies the checkpoint-loading and assistant-mask changes idempotently | After setup, source restoration, or a runtime restart |
| `!python -m py_compile train.py` | Checks the patched training script for Python syntax errors | Immediately after patching |
| `!python train.py config/train_isoliq_3m.py ...` | Runs pretraining with the documented configuration | Smoke test, full run, or resume |
| `--out_dir=...` | Sends checkpoints to a chosen folder | Keep smoke tests separate from real runs |
| `--max_iters=N` | Sets the final iteration number | Short tests or intentional training extensions |
| `--eval_interval=N` | Chooses how often validation runs | Smoke tests and evaluation scheduling |
| `--eval_iters=N` | Chooses how many batches validation averages | Fast smoke tests or steadier full evaluations |
| `--batch_size=N` | Changes sequences held in memory per micro-batch | Reduce GPU memory use |
| `--gradient_accumulation_steps=N` | Combines several micro-batches before an optimizer update | Preserve effective batch size after reducing `batch_size` |
| `--init_from=resume` | Loads model and optimizer state from the existing checkpoint | Continue an interrupted run |
| `!python -m isoliq_scripts.generate_isoliq ...` | Loads a checkpoint with the matching tokenizer and generates text | Evaluate pretrained or fine-tuned models |
| `--chat` | Wraps a prompt in the same ChatML structure used for fine-tuning | Test the fine-tuned chat checkpoint |
| `--temperature` | Controls sampling randomness; lower values are more conservative | Compare generation behavior |
| `--top_k` | Limits sampling to the most likely token candidates | Reduce very unlikely continuations |
| `--seed` | Makes a sampling run repeatable | Compare results fairly |
| `!python -m isoliq_scripts.prepare_finetune` | Parses conversations, builds splits, encodes tokens, and writes assistant-only masks | Before fine-tuning |
| `np.memmap(...)` checks | Reads token and mask files without loading them fully into RAM | Validate SFT artifacts |
| `!python -m isoliq_scripts.make_finetune_seed` | Copies pretrained weights while resetting optimizer state and counters | Once before a new fine-tuning run |
| `!cp source destination` | Copies the clean seed checkpoint into the smoke-test folder | Before the SFT smoke test |
| `for seed in [...]` | Repeats one prompt with multiple deterministic random streams | Evaluate stability instead of cherry-picking |

## Files created by this workflow

| Path inside `Isoliq Inno 1` | Purpose |
| --- | --- |
| `artifacts/tokenizer.json` | Reusable 1,280-token BPE tokenizer |
| `artifacts/pretrain_audit.json` | Corpus hash, split sizes, token counts, and preprocessing facts |
| `nanoGPT/data/isoliq/train.bin` | Encoded pretraining tokens |
| `nanoGPT/data/isoliq/val.bin` | Encoded pretraining validation tokens |
| `nanoGPT/data/isoliq/meta.pkl` | nanoGPT vocabulary metadata |
| `nanoGPT/config/train_isoliq_3m.py` | Reproducible pretraining configuration |
| `checkpoints/pretrain-smoke/ckpt.pt` | Disposable pretraining smoke-test checkpoint |
| `checkpoints/pretrain/ckpt.pt` | Main pretrained checkpoint |
| `data/finetune_data_clean.txt` | Auditable cleaned ChatML copy written by the preparation script |
| `artifacts/finetune_audit.json` | Fine-tuning duplicate, conflict, split, and token statistics |
| `nanoGPT/data/isoliq_sft/train.bin` | Encoded SFT training tokens |
| `nanoGPT/data/isoliq_sft/train_mask.bin` | Assistant-only loss mask for training |
| `nanoGPT/data/isoliq_sft/val.bin` | Encoded SFT validation tokens |
| `nanoGPT/data/isoliq_sft/val_mask.bin` | Assistant-only loss mask for validation |
| `nanoGPT/config/finetune_isoliq_3m.py` | Reproducible fine-tuning configuration |
| `checkpoints/finetune-smoke/ckpt.pt` | Disposable fine-tuning smoke-test checkpoint |
| `checkpoints/finetune/ckpt.pt` | Main fine-tuned chat checkpoint |

## Troubleshooting

### `CUDA available: False`

Select **Runtime > Change runtime type > GPU**, save the setting, and reconnect. Do not begin training on CPU unless you intentionally accept a much slower run.

### An uploaded file prints `False` or `No such file`

Check the spelling, capitalization, and Drive folder structure. Re-run Steps 1, 2, and 5. The spaces in `Isoliq Inno 1` are intentional, so keep paths quoted in shell commands.

### `ModuleNotFoundError` for `model` or `isoliq_scripts`

Run the `%cd` command from Step 4 and confirm that `pwd` would point to the nanoGPT folder. Then confirm `isoliq_scripts/__init__.py` exists.

### Checkpoint loading reports a `weights_only` error

Re-run `!python -m isoliq_scripts.patch_nanogpt`, compile `train.py`, and retry. The patch is designed to be safe when run more than once.

### CUDA runs out of memory

Use the smaller micro-batch commands in Step 11. Reduce `batch_size` while increasing `gradient_accumulation_steps` by the same factor to keep the effective batch approximately unchanged.

### Fine-tuning reports mask-length or supervised-target errors

Re-run Step 13 and its validation cell. Do not continue unless each data file has a mask of identical length and each mask contains supervised tokens.

### A runtime disconnects during training

The checkpoint should remain in Drive. Reconnect, mount Drive, reinstall packages, change into the nanoGPT folder, reapply the patch, and resume with `--init_from=resume`. Do not recreate the fine-tuning seed before a resume because that intentionally resets the run.

### A SHA-256 hash differs

You are using different data from the documented experiment. That may be intentional, but parameter behavior, token counts, run length, loss, and generated responses may differ. Record your new hashes with the results.

## Responsible use and credits

- Use only datasets you are allowed to process, and remove private or sensitive information before uploading them to Colab or Drive.
- Keep smoke-test outputs separate from real checkpoints so a test cannot overwrite useful training progress.
- Do not treat falling loss as proof of correctness, balance, factuality, or chat ability. Inspect both the data distribution and generated outputs.
- This experiment builds on Andrej Karpathy's `nanoGPT` repository and PyTorch. Keep their licenses and attribution when redistributing derived code.
- Isoliq Inno 1 is an educational experiment. Generated text may be incorrect, repetitive, biased, or unsafe.
