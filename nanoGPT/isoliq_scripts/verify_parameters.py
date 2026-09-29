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
