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
