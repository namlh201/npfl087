import json
import os
from types import SimpleNamespace

from torch import nn
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from transformers import AutoTokenizer

from block import (
    Decoder,
    GemmaDecoder,
    LlamaDecoder,
    MistralDecoder,
    QwenDecoder
)
from block import (
    Encoder,
    HubertEncoder,
    WhisperEncoder
)
from dataset import IWSLT, MUSTC

def get_config(config_files: list[str]) -> SimpleNamespace:
    merged_json = {}

    for config_file in config_files:
        with open(config_file) as f:
            data = json.load(f)
            merged_json = merged_json | data

    config = json.loads(json.dumps(merged_json), object_hook=lambda item: SimpleNamespace(**item))

    config.encoder_name = config.encoder.split('/')[1]
    config.decoder_name = config.decoder.split('/')[1]

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


def get_encoder(encoder: str, **kwargs) -> Encoder:
    if 'whisper' in encoder:
        return WhisperEncoder(encoder, **kwargs)
    elif 'hubert' in encoder:
        return HubertEncoder(encoder, **kwargs)


def get_decoder(decoder: str, vocab_size: int, lora_params: SimpleNamespace=None, debug=False, **kwargs) -> Decoder:
    # if decoder == 'gpt-2':
    #     return GPT2Decoder(vocab_size)
    if 'gemma' in decoder:
        return GemmaDecoder(vocab_size, decoder, lora_params=lora_params, debug=debug, **kwargs)
        # return GemmaDecoder(vocab_size, init_lora=init, mbr_decode=mbr_decode)
    elif 'llama' in decoder.lower():
        return LlamaDecoder(vocab_size, decoder, lora_params=lora_params, debug=debug, **kwargs)
    elif 'mistral' in decoder.lower():
        return MistralDecoder(vocab_size, decoder, lora_params=lora_params, debug=debug, **kwargs)
    elif 'qwen' in decoder.lower():
        return QwenDecoder(vocab_size, decoder, lora_params=lora_params, debug=debug, **kwargs)

def get_tokenizer(decoder: str, **kwargs) -> tuple[nn.Module, dict[str, int]]:
    SPECIAL_TOKENS = ['<|audio|>', '<|transcript|>', '<|translation|>']

    # if decoder == 'gpt-2':
    #     tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    # if 'gemma' in decoder:
    #     tokenizer = GemmaTokenizer.from_pretrained(decoder, token=os.environ['HF_TOKEN'])
    # elif 'llama' in decoder.lower():
    #     tokenizer = LlamaTokenizer.from_pretrained(decoder, token=os.environ['HF_TOKEN'])
    # elif 'mistral' in decoder.lower():
    tokenizer = AutoTokenizer.from_pretrained(decoder, add_bos_token=False, token=os.environ['HF_TOKEN'], **kwargs)

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

    # if not tokenizer.pad_token_id:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     # tokenizer.pad_token_id = tokenizer.eos_token_id

    special_token_ids = {
        'unk_token': tokenizer.unk_token_id,
        'bos_token': tokenizer.bos_token_id,
        'eos_token': tokenizer.eos_token_id,
        'pad_token': tokenizer.pad_token_id,
        'audio_token': added_vocab['<|audio|>'],
        'transcript_token': added_vocab['<|transcript|>'],
        'translation_token': added_vocab['<|translation|>']
    }

    special_tokens = {
        'unk_token': tokenizer.unk_token,
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token,
        'pad_token': tokenizer.pad_token,
        'audio_token': '<|audio|>',
        'transcript_token': '<|transcript|>',
        'translation_token': '<|translation|>'
    }

    # tokenizer.padding_side = 'right'

    # tokenizer.add_bos_token = False

    # print(special_token_ids)

    # print(tokenizer.get_added_vocab())

    return tokenizer, special_token_ids, special_tokens