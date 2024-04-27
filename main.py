import os

import torch.utils
import torch.utils.data

os.environ['HF_HOME'] = os.getcwd() + '/checkpoints'

# from datasets import Dataset, Audio
# import numpy as np
import torch
from torch import nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH as LibriSpeech
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from block import Encoder, GPT2Decoder, get_tokenizer
from data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SAMPLE_RATE = 16_000
DATA_DIR = os.path.join(os.getcwd(), 'data')

BATCH_SIZE = 1
NUM_EPOCHS = 10

def train(
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    dev_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for epoch in range(num_epochs):
        mean_loss = 0.0

        print(f'Epoch #{epoch + 1}:')
        for audio_feats, transcripts in tqdm(train_loader):
            loss = train_step(
                encoder,
                decoder,
                tokenizer,
                audio_feats=audio_feats,
                transcripts=transcripts,
                translations=transcripts,
                special_token_ids=special_token_ids
            )

            mean_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        mean_loss = mean_loss / len(train_loader)

        print(f'Loss = {mean_loss}')


def train_step(
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: AutoTokenizer,
    audio_feats: torch.Tensor,
    transcripts: list[str],
    translations: list[str],
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    audio_tok_id, transcript_tok_id, translation_tok_id, eot_tok_id = special_token_ids
    embed = decoder.get_input_embeddings()

    audio_feats = audio_feats.to(device)

    audio_hidden_feats = encoder(audio_feats)

    transcripts = list(
        map(
            lambda transcript: tokenizer(transcript)['input_ids'],
            transcripts
        )
    )
    padded_transcripts = nn.utils.rnn.pad_sequence(
        [torch.tensor(transcript) for transcript in transcripts],
        batch_first=True
    ).to(device)
    embeded_transcripts = embed(padded_transcripts).view((BATCH_SIZE, padded_transcripts.shape[1], -1))

    translations = list(
        map(
            lambda translation: tokenizer(translation)['input_ids'],
            translations
        )
    )
    padded_translations = nn.utils.rnn.pad_sequence(
        [torch.tensor(translation) for translation in translations],
        batch_first=True
    ).to(device)
    embeded_translations = embed(padded_translations).view((BATCH_SIZE, padded_translations.shape[1], -1))

    embeded_audio_token = embed(audio_tok_id).view((BATCH_SIZE, 1, -1))
    embeded_transcript_token = embed(transcript_tok_id).view((BATCH_SIZE, 1, -1))
    embeded_translation_token = embed(translation_tok_id).view((BATCH_SIZE, 1, -1))
    embeded_eot_token = embed(eot_tok_id).view((BATCH_SIZE, 1, -1))

    input_feats = torch.cat(
        (
            embeded_audio_token, audio_hidden_feats, \
            embeded_transcript_token, embeded_transcripts, \
            embeded_translation_token, embeded_translations, \
            embeded_eot_token
        ),
        dim=1
    )
    input_feats = input_feats.to(device)

    translation_attention_masks = [
        [1] * input_feats.shape[0]
    ]
    translation_labels = [
        [-100] * (embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + \
        # embeded_transcripts.shape[1] + embeded_translation_token.shape[1]) + \
        transcripts[0] + \
        [translation_tok_id] + \
        translations[0] + \
        [eot_tok_id]
    ]
    translation_attention_masks = torch.tensor(translation_attention_masks).view((BATCH_SIZE, -1)).to(device)
    translation_labels = torch.tensor(translation_labels).view((BATCH_SIZE, -1)).to(device)

    translation_output = decoder(inputs_embeds=input_feats, attention_mask=translation_attention_masks, labels=translation_labels)

    translation_loss = translation_output.loss

    loss = translation_loss

    return loss


# def eval_one(
#     encoder: nn.Module,
#     decoder: nn.Module,
#     tokenizer: AutoTokenizer,
#     audio_feats: torch.Tensor,
#     transcripts: list[str],
#     translations: list[str],
#     special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
# ) -> None:
#     pass


def main():
    librispeech_train = LibriSpeech(DATA_DIR, url='train-clean-100', download=True)
    # librispeech_dev = LibriSpeech(DATA_DIR, url='dev-clean', download=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
    # sampling_rate = feature_extractor.sampling_rate

    tokenizer = get_tokenizer('gpt-2')
    # print(tokenizer)

    decoder = GPT2Decoder(tokenizer).to(device)
    dec_hidden_size = decoder.get_hidden_size()

    encoder = Encoder(dec_hidden_size).to(device)

    train_loader = DataLoader(librispeech_train, feature_extractor, batch_size=BATCH_SIZE)
    # dev_loader = DataLoader(librispeech_dev, feature_extractor, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(
        list(encoder.project.parameters()) + list(decoder.parameters()),
        lr=0.001,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 10, NUM_EPOCHS * len(train_loader))

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    encoder.train()
    decoder.train()

    # embed = decoder.get_input_embeddings()

    # i = 0

    audio_tok_id = tokenizer('<|audio|>')['input_ids'][0]
    audio_tok_id = torch.full((BATCH_SIZE, ), audio_tok_id).to(device)

    transcript_tok_id = tokenizer('<|transcript|>')['input_ids'][0]
    transcript_tok_id = torch.full((BATCH_SIZE, ), transcript_tok_id).to(device)

    translation_tok_id = tokenizer('<|translation|>')['input_ids'][0]
    translation_tok_id = torch.full((BATCH_SIZE, ), translation_tok_id).to(device)

    eot_tok_id = tokenizer('<|endoftext|>')['input_ids'][0]
    eot_tok_id = torch.full((BATCH_SIZE, ), eot_tok_id).to(device)

    train(
        encoder,
        decoder,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_loader=train_loader,
        dev_loader=None,
        num_epochs=NUM_EPOCHS,
        special_token_ids=(audio_tok_id, transcript_tok_id, translation_tok_id, eot_tok_id)
    )

    os.makedirs('models', exist_ok=True)
    torch.save(encoder.project.state_dict(), os.path.join('models', 'hubert_to_gpt2_projection.pth'))
    torch.save(decoder.state_dict(), os.path.join('models', 'gpt2.pth'))

    # for audio_feats, transcripts in tqdm(loader):
    #     # print(waveforms)

    #     # feats = feature_extractor(waveforms[0], sampling_rate=sampling_rate, return_tensors='pt').input_values

    #     # feats = feats.to(device)
    #     # print(feats, feats.shape)

    #     audio_feats = audio_feats.to(device)
    #     # audio_feat_masks = audio_feat_masks.to(device)

    #     audio_hidden_feats = encoder(audio_feats)
    #     # audio_hidden_feats = encoder(audio_feats, audio_feat_masks)

    #     # print(ctc_feats, ctc_feats.shape)

    #     transcripts = list(
    #         map(
    #             lambda transcript: tokenizer(transcript)['input_ids'],
    #             transcripts
    #         )
    #     )
    #     translations = transcripts
    #     padded_transcripts = nn.utils.rnn.pad_sequence(
    #         [torch.tensor(transcript) for transcript in transcripts],
    #         batch_first=True
    #     ).to(device)
    #     embeded_transcripts = embed(padded_transcripts).view((BATCH_SIZE, padded_transcripts.shape[1], -1))
    #     embeded_translations = embeded_transcripts

    #     # audio_tok_id = tokenizer('<|audio|>')['input_ids'][0]
    #     # audio_tok_id = torch.full((BATCH_SIZE, ), audio_tok_id).to(device)
    #     embeded_audio_token = embed(audio_tok_id).view((BATCH_SIZE, 1, -1))

    #     # transcript_tok_id = tokenizer('<|transcript|>')['input_ids'][0]
    #     # transcript_tok_id = torch.full((BATCH_SIZE, ), transcript_tok_id).to(device)
    #     embeded_transcript_token = embed(transcript_tok_id).view((BATCH_SIZE, 1, -1))

    #     # translation_tok_id = tokenizer('<|translation|>')['input_ids'][0]
    #     # translation_tok_id = torch.full((BATCH_SIZE, ), translation_tok_id).to(device)
    #     embeded_translation_token = embed(translation_tok_id).view((BATCH_SIZE, 1, -1))

    #     # eot_tok_id = tokenizer('<|endoftext|>')['input_ids'][0]
    #     # eot_tok_id = torch.full((BATCH_SIZE, ), eot_tok_id).to(device)
    #     embeded_eot_token = embed(eot_tok_id).view((BATCH_SIZE, 1, -1))

    #     # print(embeded_audio_token.shape, audio_hidden_feats.shape, embeded_transcripts.shape)

    #     hidden_feats = torch.cat(
    #         (
    #             embeded_audio_token, audio_hidden_feats, \
    #             embeded_transcript_token, embeded_transcripts, \
    #             embeded_translation_token, embeded_translations, \
    #             embeded_eot_token
    #         ),
    #         dim=1
    #     )

    #     # hidden_feats = hidden_feats.squeeze(dim=0)
    #     # hidden_feats = hidden_feats.bfloat16()
    #     hidden_feats = hidden_feats.to(device)

    #     # asr_attention_masks = [
    #     #     [1] * (embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1] + embeded_transcripts.shape[1]) + \
    #     #     [0] * (embeded_translation_token.shape[1] + embeded_translations.shape[1] + embeded_eot_token.shape[1])
    #     # ]
    #     # asr_labels = [
    #     #     [-100] * (embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + \
    #     #     transcripts[0] + \
    #     #     [translation_tok_id] + \
    #     #     [-100] * (embeded_translations.shape[1] + embeded_eot_token.shape[1])
    #     # ]
    #     # asr_attention_masks = torch.tensor(asr_attention_masks).view((BATCH_SIZE, -1)).to(device)
    #     # asr_labels = torch.tensor(asr_labels).view((BATCH_SIZE, -1)).to(device)

    #     translation_attention_masks = [
    #         [1] * hidden_feats.shape[0]
    #     ]
    #     translation_labels = [
    #         [-100] * (embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + \
    #         # embeded_transcripts.shape[1] + embeded_translation_token.shape[1]) + \
    #         transcripts[0] + \
    #         [translation_tok_id] + \
    #         translations[0] + \
    #         [eot_tok_id]
    #     ]
    #     translation_attention_masks = torch.tensor(translation_attention_masks).view((BATCH_SIZE, -1)).to(device)
    #     translation_labels = torch.tensor(translation_labels).view((BATCH_SIZE, -1)).to(device)

    #     # asr_labels = asr_masks * 

    #     # cat_embeds = torch.cat((embeded_aud, ctc_hidden, embeded_prompt, embeded_trans), dim=1).bfloat16()

    #     # labels = [[-100] * (len(aud_tok_id[0]) + len(ctc_hidden[0]) + len(prompt_tok_id[0]))]
    #     # labels = torch.tensor(labels).to(device)
    #     # labels = torch.cat((labels, trans), dim=-1)

    #     # print(cat_embeds.shape, labels.shape)

    #     # print(hidden_feats.shape)

    #     # asr_output = decoder(inputs_embeds=hidden_feats, attention_mask=asr_attention_masks, labels=asr_labels)
    #     translation_output = decoder(inputs_embeds=hidden_feats, attention_mask=translation_attention_masks, labels=translation_labels)

    #     # asr_loss = asr_output.loss
    #     translation_loss = translation_output.loss

    #     loss = translation_loss
    #     print(loss)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     lr_scheduler.step()

        # i += 1




if __name__ == '__main__':
    main()