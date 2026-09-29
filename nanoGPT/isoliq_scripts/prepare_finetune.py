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
