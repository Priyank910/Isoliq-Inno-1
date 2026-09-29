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
