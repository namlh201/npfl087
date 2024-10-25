import os
from pathlib import Path

import torch
import torch.utils.data
from torchaudio.datasets.utils import _load_waveform

class MUSTC(torch.utils.data.Dataset):
    _SAMPLE_RATE = 16_000
    _FOLDER = 'MUSTC'

    _ext_txt = '.txt'
    _ext_audio = '.wav'

    def __init__(
        self,
        root: str,
        direction: str,
        subset: str,
        folder: str=_FOLDER,
    ) -> None:
        root = os.fspath(root)

        self.subset = subset

        self.path = os.path.join(root, folder, direction, subset)

        self.walker = sorted(str(p.stem) for p in Path(self.path).glob("*/*" + self._ext_audio))

    def __len__(self) -> int:
        return len(self.walker)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, str, str, dict[str, str]]:
        file_id = self.walker[index]
        talk_id, chunk_id = file_id.split('-')

        audio_path = os.path.join(talk_id, file_id + self._ext_audio)
        waveform = _load_waveform(self.path, audio_path, self._SAMPLE_RATE)

        if self.subset == 'test' or 'tst' in self.subset:
            src, tgt = '', ''
        else:
            transcript_path = os.path.join(talk_id, file_id + self._ext_txt)

            with open(os.path.join(self.path, transcript_path)) as f:
                transcripts = f.readline().strip()

                src, tgt = transcripts.split('\t')

        misc = {
            'talk_id': talk_id,
            'chunk_id': chunk_id
        }

        return waveform, src, tgt, misc

class IWSLT(torch.utils.data.Dataset):
    _SAMPLE_RATE = 16_000
    _FOLDER = 'iwslt'

    _ext_txt = '.txt'
    _ext_audio = '.wav'

    def __init__(
        self,
        root: str,
        direction: str,
        subset: str,
        folder: str=_FOLDER,
    ) -> None:
        root = os.fspath(root)

        self.subset = subset

        self.path = os.path.join(root, folder, direction, subset)

        self.walker = sorted(str(p.stem) for p in Path(self.path).glob("*/*" + self._ext_audio))

    def __len__(self) -> int:
        return len(self.walker)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, str, str, dict[str, str]]:
        file_id = self.walker[index]
        talk_id, chunk_id = file_id.split('-')

        audio_path = os.path.join(talk_id, file_id + self._ext_audio)
        waveform = _load_waveform(self.path, audio_path, self._SAMPLE_RATE)

        if self.subset == 'test' or 'tst' in self.subset:
            src, tgt = '', ''
        else:
            transcript_path = os.path.join(talk_id, file_id + self._ext_txt)

            with open(os.path.join(self.path, transcript_path)) as f:
                transcripts = f.readline().strip()

                src, tgt = transcripts.split('\t')

        misc = {
            'talk_id': talk_id,
            'chunk_id': chunk_id
        }

        return waveform, src, tgt, misc
