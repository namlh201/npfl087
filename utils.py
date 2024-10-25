import json
import os
from types import SimpleNamespace

from torch import nn
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from transformers import AutoTokenizer, GemmaTokenizer, GPT2Tokenizer, LlamaTokenizer

from block import Decoder, GPT2Decoder, GemmaDecoder, LlamaDecoder, MistralDecoder
from dataset import IWSLT, MUSTC

def get_config(config_file: str) -> SimpleNamespace:
    with open(config_file) as f:
        config = SimpleNamespace(**json.load(f))

    return config

def get_dataset(name: str, direction: str, subset: str, root: str=None) -> Dataset:
    _DATA_DIR = os.path.join(os.getcwd(), 'data')
    _VALID = {
        'MUSTC': {
            'direction': ['en-cs', 'en-vi', 'en-de', 'en-ja', 'en-zh'],
            'subset': ['train', 'dev', 'test', 'tst-COMMON_v3', 'tst-HE_v3', 'tst-COMMON_v2', 'tst-HE_v2'],
        },
        'iwslt': {
            'direction': ['en-de', 'en-ja', 'en-zh'],
            'subset': ['tst2019', 'tst2020', 'tst2021', 'tst2022'],
        },
        'LIBRISPEECH': {
            'direction': ['en-en'],
            'subset': [
                'train-clean-100', 'train-clean-360', 'train-clean-500',
                'dev-clean', 'dev-other',
                'test-clean', 'test-other'
            ],
        }
        
    }

    assert name in _VALID, f"Invalid dataset {name}. Supported datasets: {list(_VALID.keys())}"
    assert direction in _VALID[name]['direction'], f"Invalid direction {direction}. Available directions for dataset {name}: {_VALID[name]['direction']}"
    assert subset in _VALID[name]['subset'], f"Invalid subset {subset}. Available subsets for dataset {name}: {_VALID[name]['subset']}"

    if not root:
        root = _DATA_DIR

    if name == 'MUSTC':
        return MUSTC(root, direction=direction, subset=subset)
    elif name == 'iwslt':
        return IWSLT(root, direction=direction, subset=subset)
    elif name == 'LIBRISPEECH':
        return LIBRISPEECH(root, url=subset, download=True)

def get_decoder(decoder: str, vocab_size: int, init: bool=True) -> Decoder:
    if decoder == 'gpt-2':
        return GPT2Decoder(vocab_size)
    elif 'gemma' in decoder:
        return GemmaDecoder(vocab_size, decoder, init_lora=init)
        # return GemmaDecoder(vocab_size, init_lora=init, mbr_decode=mbr_decode)
    elif 'llama' in decoder.lower():
        return LlamaDecoder(vocab_size, decoder, init_lora=init)
    elif 'mistral' in decoder.lower():
        return MistralDecoder(vocab_size, decoder, init_lora=init)

def get_tokenizer(decoder: str='gpt-2') -> tuple[nn.Module, dict[str, int]]:
    SPECIAL_TOKENS = ['<|audio|>', '<|transcript|>', '<|translation|>']

    if decoder == 'gpt-2':
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    elif 'gemma' in decoder:
        tokenizer = GemmaTokenizer.from_pretrained(f'google/{decoder}', token=os.environ['HF_TOKEN'])
    elif 'llama' in decoder.lower():
        tokenizer = LlamaTokenizer.from_pretrained(f'meta-llama/{decoder}', token=os.environ['HF_TOKEN'])
    elif 'mistral' in decoder.lower():
        tokenizer = AutoTokenizer.from_pretrained(f'mistralai/{decoder}', token=os.environ['HF_TOKEN'])

    tokenizer.add_special_tokens({
        # 'unk_token': '<|endoftext|>',
        # 'bos_token': '<|endoftext|>',
        # 'eos_token': '<|endoftext|>',
        # 'pad_token': '<|endoftext|>',
        # 'unk_token': tokenizer.eos_token,
        # 'bos_token': tokenizer.eos_token,
        # 'eos_token': tokenizer.eos_token,
        # 'pad_token': tokenizer.eos_token,
        'additional_special_tokens': SPECIAL_TOKENS
    })

    added_vocab = tokenizer.get_added_vocab()

    special_token_ids = {
        'unk_token': tokenizer.unk_token_id,
        'bos_token': tokenizer.bos_token_id,
        'eos_token': tokenizer.eos_token_id,
        'pad_token': tokenizer.pad_token_id,
        'audio_token': added_vocab['<|audio|>'],
        'transcript_token': added_vocab['<|transcript|>'],
        'translation_token': added_vocab['<|translation|>']
    }

    tokenizer.add_bos_token = False

    # print(special_token_ids)

    # print(tokenizer.get_added_vocab())

    return tokenizer, special_token_ids