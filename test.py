from utils import *

data_dir = "./data"
num_examples=None
num_steps = 35

raw_file = read_data_nmt('fra.txt')
text = preprocess_nmt(raw_file)
source, target = tokenize_nmt(text, num_examples)
src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
print(src_array.shape, src_valid_len.shape, tgt_array.shape, tgt_valid_len.shape)