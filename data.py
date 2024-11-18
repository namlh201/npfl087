from typing import Callable

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

        self.dataset_name = type(dataset).__name__
        self.feature_extractor_name = type(feature_extractor).__name__

        self.feature_extractor = feature_extractor
        self.sampling_rate = self.feature_extractor.sampling_rate

        # print(self.feature_extractor_name)

        collate_fn = collate_fn if collate_fn else self.collate_fn

        super().__init__(dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn)

    def collate_fn(self, batch):
        if self.dataset_name == 'LIBRISPEECH':
            waveforms = list(map(lambda data: data[0][0], batch))
            transcripts = list(map(lambda data: data[2], batch))
            translations = transcripts
            misc = []
        elif self.dataset_name == 'MUSTC':
            waveforms = list(map(lambda data: data[0], batch))
            transcripts = list(map(lambda data: data[1], batch))
            translations = list(map(lambda data: data[2], batch))
            misc = list(map(lambda data: data[3], batch))
        elif self.dataset_name == 'IWSLT':
            waveforms = list(map(lambda data: data[0], batch))
            transcripts = list(map(lambda data: data[1], batch))
            translations = list(map(lambda data: data[2], batch))
            misc = list(map(lambda data: data[3], batch))

        if self.feature_extractor_name == 'SeamlessM4TFeatureExtractor':
            audio_feats = list(
                map(
                    lambda waveform: 
                        self.feature_extractor(waveform, sampling_rate=self.sampling_rate, return_tensors='pt').input_features.squeeze(),
                    waveforms
                )
            )
        elif self.feature_extractor_name == 'WhisperFeatureExtractor':
            audio_feats = list(
                map(
                    lambda waveform: 
                        self.feature_extractor(waveform[0], sampling_rate=self.sampling_rate, return_tensors='pt').input_features.squeeze(),
                    waveforms
                )
            )    
        else:
            audio_feats = list(
                map(
                    lambda waveform: 
                        self.feature_extractor(waveform, sampling_rate=self.sampling_rate, return_tensors='pt').input_values.squeeze(),
                    waveforms
                )
            )
        audio_feats = nn.utils.rnn.pad_sequence(audio_feats, batch_first=True)

        return audio_feats, transcripts, translations, misc