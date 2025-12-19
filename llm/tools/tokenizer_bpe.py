import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def train_or_load_tokenizer(text_iterable, save_path: str, vocab_size: int = 8000):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if os.path.exists(save_path):
        tok = Tokenizer.from_file(save_path)
        return tok

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tok.train_from_iterator(text_iterable, trainer=trainer)
    tok.save(save_path)
    return tok

def encode(tok: Tokenizer, text: str):
    return tok.encode(text).ids

def decode(tok: Tokenizer, ids):
    return tok.decode(ids)
