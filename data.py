from typing import Callable

import torch
from torch import nn
import torch.utils.data
from transformers import AutoFeatureExtractor

class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        dataset: torch.utils.data.Dataset,
        feature_extractor: AutoFeatureExtractor,
        batch_size: int,
        shuffle: bool=False,
        num_workers: int=1,
        collate_fn: Callable=None,
    ) -> None:
        # self.encoder = encoder
        # self.tokenizer = encoder.tokenizer

        self.feature_extractor = feature_extractor
        self.sampling_rate = self.feature_extractor.sampling_rate


        collate_fn = collate_fn if collate_fn else self.collate_fn

        super().__init__(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def collate_fn(self, batch):
        waveforms = list(map(lambda data: data[0][0], batch))
        transcripts = list(map(lambda data: data[2], batch))

        audio_feats = list(
            map(
                lambda waveform: 
                    self.feature_extractor(waveform, sampling_rate=self.sampling_rate, return_tensors='pt').input_values.squeeze(),
                waveforms
            )
        )
        audio_feats = nn.utils.rnn.pad_sequence(audio_feats, batch_first=True)

        # audio_feat_lengths = torch.count_nonzero(audio_feats, dim=-1)

        # audio_feat_masks = audio_feats != 0

        # audio_hidden_feats, _ = self.encoder(audio_feats, audio_feat_masks)

        return audio_feats, transcripts

        # return (audio_feats, audio_feat_masks), transcripts