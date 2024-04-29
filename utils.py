import os

from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH

from dataset import MUSTC

def get_dataset(name: str, direction: str, subset: str, root: str=None) -> Dataset:
    _DATA_DIR = os.path.join(os.getcwd(), 'data')
    _VALID = {
        'MUSTC': {
            'direction': ['en-cs'],
            'subset': ['train', 'dev'],
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
    elif name == 'LIBRISPEECH':
        return LIBRISPEECH(root, url=subset, download=True)