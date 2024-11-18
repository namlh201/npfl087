import os
import argparse
from datetime import datetime
import warnings
from types import SimpleNamespace
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
parser.add_argument('--steps', default=100000, type=int, help='Number of steps.')
parser.add_argument('--config', type=lambda args: args.split(','), required=True, help='Config files.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')
parser.add_argument('--debug', default=False, action='store_true', help='Debug.')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
os.environ['HF_HOME'] = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'

from dotenv import load_dotenv
import numpy as np
import torch
from torch import nn
import torch.utils
import torch.utils.data
# from torchaudio.functional import edit_distance
from transformers import AutoFeatureExtractor, PreTrainedModel, PreTrainedTokenizer
from transformers import get_cosine_schedule_with_warmup
from transformers import logging
from tqdm import tqdm
import wandb

load_dotenv()
logging.set_verbosity_error()

from block import LengthAdapter, Projection
from data import DataLoader
from utils import get_config, get_dataset, get_decoder, get_encoder, get_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(
    encoder: PreTrainedModel,
    projection: nn.Module,
    decoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    dev_loader: torch.utils.data.DataLoader,
    num_steps: int,
    # special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    special_tokens: tuple[str, str],
) -> None:
    # encoder.train()
    # projection.train()
    # decoder.train()

    # grad_accum_int = 4

    step = 1

    mean_loss = 0.0

    checkpoint_step = 10000 if num_steps == 100000 else 1000

    # for epoch in range(num_epochs):
    while step < num_steps:
        # mean_loss = 0.0

        # valid_size = len(train_loader)

        # step = 0

        # print(f'Epoch #{epoch + 1}:')
        data_and_progress = tqdm(train_loader, leave=False)
        for audio_feats, transcripts, translations, _ in data_and_progress:
            # if i == 100:
            #     break

            try:
                # if args.randomized_factor and np.random.uniform() < args.randomized_factor:
                #     loss = train_step_randomized(
                #         # encoder,
                #         # projection,
                #         decoder,
                #         tokenizer,
                #         # audio_feats=audio_feats,
                #         transcripts=transcripts,
                #         translations=translations,
                #         special_token_ids=special_token_ids
                #     )
                # else:
                loss = train_step(
                    encoder,
                    projection,
                    decoder,
                    tokenizer,
                    audio_feats=audio_feats,
                    transcripts=transcripts,
                    translations=translations,
                    special_token_ids=special_token_ids,
                    special_tokens=special_tokens
                )

                mean_loss += loss.item()

                # loss = loss / grad_accum_int

                # loss.backward()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                data_and_progress.set_description(f'loss = {loss.item()}', refresh=False)

                # if ((i + 1) % grad_accum_int == 0) or (i + 1 == len(train_loader)):
                #     optimizer.step()
                #     lr_scheduler.step()
                #     optimizer.zero_grad()

                step += 1
            except Exception as e:
                # valid_size -= 1
                # step = num_steps
                # break

                data_and_progress.set_description(f'loss = inf', refresh=False)

                if args.debug:
                    # print(valid_size)
                    print(transcripts)
                    print(translations)
                    print(e)
                    print()
                continue
                # break

            # i += 1

            if step % checkpoint_step == 0:
                mean_loss = mean_loss / checkpoint_step

                wandb.log({
                    'step': step,
                    'loss': mean_loss
                })

                print(f'Loss = {mean_loss}')

                mean_loss = 0.0

        # os.makedirs(os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}'), exist_ok=True)
        # os.makedirs(os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder'), exist_ok=True)

        # torch.save(
        #     projection.state_dict(),
        #     os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', f'hubert_to_{args.decoder}_projection_e{epoch + 1}.pth')
        # )

        # if args.decoder == 'gpt-2':
        #     decoder.save_pretrained(os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder', f'{args.decoder}.pth'))
        # elif 'gemma' in args.decoder:
        #     decoder.save_pretrained(os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder'))

        # decoder.save_pretrained(
        #     os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder', f'{args.decoder}_e{epoch + 1}.pth')
        # )
        # torch.save(
        #     decoder.state_dict(),
        #     os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder', f'{args.decoder}_e{epoch + 1}.pth')
        # )

def train_step(
    encoder: PreTrainedModel,
    projection: nn.Module,
    decoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    audio_feats: torch.Tensor,
    transcripts: list[str],
    translations: list[str],
    # special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    special_tokens: tuple[str, str],
) -> torch.Tensor:
    # bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id = special_token_ids
    bos_tok_id, audio_tok_id, transcript_tok_id = special_token_ids
    translation_tok, eos_tok = special_tokens
    embed = decoder.get_input_embeddings()

    audio_feats = audio_feats.to(device)

    # audio_attention_masks = torch.ones_like(audio_feats).to(device)
    audio_hidden_feats = encoder(audio_feats) #, attention_mask=audio_attention_masks)
    audio_hidden_feats = projection(audio_hidden_feats)

    attention_masks = []
    transcripts_and_translations = []

    for transcript, translation in zip(transcripts, translations):
        merged = f'{transcript} {translation_tok} {translation} {eos_tok}'
        merged = tokenizer(merged, add_special_tokens=False)

        attention_masks.append(merged['attention_mask'])
        transcripts_and_translations.append(merged['input_ids'])

        # transcripts_and_translations.append(merged)

    # attention_masks = nn.utils.rnn.pad_sequence(
    #     [torch.tensor(attention_mask) for attention_mask in attention_masks],
    #     batch_first=True
    # ).to(device)
    embeded_transcripts_and_translations = nn.utils.rnn.pad_sequence(
        [torch.tensor(merged) for merged in transcripts_and_translations],
        batch_first=True
    ).to(device)

    # embeded_transcripts_and_translations = tokenizer(transcripts_and_translations, padding=True, add_special_tokens=False, return_tensors='pt')
    # attention_masks = embeded_transcripts_and_translations['attention_mask']
    # embeded_transcripts_and_translations = embeded_transcripts_and_translations['input_ids']

    embeded_transcripts_and_translations = embed(embeded_transcripts_and_translations).view((args.batch_size, embeded_transcripts_and_translations.shape[1], -1))

    # transcripts = list(
    #     map(
    #         lambda transcript: tokenizer(transcript)['input_ids'],
    #         transcripts
    #     )
    # )
    # padded_transcripts = nn.utils.rnn.pad_sequence(
    #     [torch.tensor(transcript) for transcript in transcripts],
    #     batch_first=True
    # ).to(device)
    # embeded_transcripts = embed(padded_transcripts).view((args.batch_size, padded_transcripts.shape[1], -1))

    # translations = list(
    #     map(
    #         lambda translation: tokenizer(translation)['input_ids'],
    #         translations
    #     )
    # )
    # padded_translations = nn.utils.rnn.pad_sequence(
    #     [torch.tensor(translation) for translation in translations],
    #     batch_first=True
    # ).to(device)
    # embeded_translations = embed(padded_translations).view((args.batch_size, padded_translations.shape[1], -1))

    embeded_bos_token = embed(bos_tok_id).view((args.batch_size, 1, -1))
    embeded_audio_token = embed(audio_tok_id).view((args.batch_size, 1, -1))
    embeded_transcript_token = embed(transcript_tok_id).view((args.batch_size, 1, -1))
    # embeded_translation_token = embed(translation_tok_id).view((args.batch_size, 1, -1))
    # embeded_eos_token = embed(eos_tok_id).view((args.batch_size, 1, -1))

    # input_feats = torch.cat(
    #     (
    #         embeded_bos_token, \
    #         embeded_audio_token, audio_hidden_feats, \
    #         embeded_transcript_token, embeded_transcripts, \
    #         embeded_translation_token, embeded_translations, \
    #         embeded_eos_token
    #     ),
    #     dim=1
    # )
    input_feats = torch.cat(
        (
            embeded_bos_token, \
            embeded_audio_token, audio_hidden_feats, \
            embeded_transcript_token, \
            embeded_transcripts_and_translations
        ),
        dim=1
    )
    input_feats = input_feats.bfloat16() if config.decoder != 'gpt-2' else input_feats
    input_feats = input_feats.to(device)

    label_masks = nn.utils.rnn.pad_sequence(
        [
            torch.tensor(
                [0] * (embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + attention_mask
            ) for attention_mask in attention_masks
        ],
        batch_first=True
    ).to(device)

    attention_masks = nn.utils.rnn.pad_sequence(
        [
            torch.tensor(
                [1] * (embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + attention_mask
            ) for attention_mask in attention_masks
        ],
        batch_first=True
    ).to(device)

    labels_transcripts_and_translations = nn.utils.rnn.pad_sequence(
        [
            torch.tensor(
                [0] * (embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + merged
            ) for merged in transcripts_and_translations
        ],
        batch_first=True
    ).to(device)

    labels = (-100) * (1 - label_masks) + labels_transcripts_and_translations
    labels = labels.to(device)

    del label_masks

    # print(input_feats.shape, attention_masks.shape, labels.shape)
    # print(attention_masks)
    # print(labels)

    # translation_attention_masks = [
    #     [1] * input_feats.shape[0]
    # ]
    # translation_labels = [
    #     [-100] * (embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + \
    #     # embeded_transcripts.shape[1] + embeded_translation_token.shape[1]) + \
    #     transcripts[0] + \
    #     [translation_tok_id.tolist()[0]] + \
    #     translations[0] + \
    #     [eos_tok_id.tolist()[0]]
    # ]

    # print(translation_labels)
    # translation_attention_masks = torch.tensor(translation_attention_masks).view((args.batch_size, -1)).to(device)
    # translation_labels = torch.tensor(translation_labels).view((args.batch_size, -1)).to(device)

    translation_output = decoder(
        inputs_embeds=input_feats,
        attention_mask=attention_masks,
        labels=labels
    )

    translation_loss = translation_output.loss

    loss = translation_loss

    # beg = embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]

    # print(loss)
    # print(translation_labels)
    # print(tokenizer.decode(translation_labels.tolist()[0][beg:]))

    return loss

def main(args: argparse.Namespace, config: SimpleNamespace):
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.encoder)

    tokenizer, special_token_ids, special_tokens = get_tokenizer(config.decoder)
    # print(tokenizer)

    encoder = get_encoder(config.encoder).to(device)
    enc_hidden_size = encoder.get_hidden_size()

    decoder = get_decoder(config.decoder, len(tokenizer), lora_params=config.lora_params).to(device)
    dec_hidden_size = decoder.get_hidden_size()

    if config.length_adapter:
        projection = nn.Sequential(
            LengthAdapter(args.batch_size),
            Projection(enc_hidden_size, dec_hidden_size)
        ).to(device)
    else:
        projection = Projection(enc_hidden_size, dec_hidden_size).to(device)

    bos_tok_id = special_token_ids['bos_token']
    bos_tok_id = torch.full((args.batch_size, ), bos_tok_id).to(device)

    audio_tok_id = special_token_ids['audio_token']
    audio_tok_id = torch.full((args.batch_size, ), audio_tok_id).to(device)

    transcript_tok_id = special_token_ids['transcript_token']
    transcript_tok_id = torch.full((args.batch_size, ), transcript_tok_id).to(device)

    # translation_tok_id = special_token_ids['translation_token']
    # translation_tok_id = torch.full((args.batch_size, ), translation_tok_id).to(device)

    # eos_tok_id = special_token_ids['eos_token']
    # eos_tok_id = torch.full((args.batch_size, ), eos_tok_id).to(device)

    # transcript_tok = special_tokens['transcript_token']
    translation_tok = special_tokens['translation_token']
    eos_tok = special_tokens['eos_token']

    train_data = get_dataset(name=config.dataset, direction=config.direction, subset=config.train_subset, root=args.data_dir)
    train_loader = DataLoader(train_data, feature_extractor, batch_size=args.batch_size)

    # print(len(train_loader))

    print(feature_extractor)
    # print(tokenizer)
    print(encoder)
    print(projection)
    print(decoder)


    optimizer = torch.optim.AdamW(
        list(projection.parameters()) + list(decoder.parameters()),
        lr=config.train_params.lr,
    )
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, args.epochs * len(train_loader))
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 100, args.steps)

    encoder.train()
    projection.train()
    decoder.train()

    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('models', f'{config.direction}'), exist_ok=True)
    os.makedirs(os.path.join('models', f'{config.direction}', 'decoder'), exist_ok=True)

    train(
        encoder,
        projection,
        decoder,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_loader=train_loader,
        dev_loader=None,
        # num_epochs=args.epochs,
        num_steps=args.steps,
        # special_token_ids=(bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id),
        special_token_ids=(bos_tok_id, audio_tok_id, transcript_tok_id),
        special_tokens=(translation_tok, eos_tok)
    )

    torch.save(
        projection.state_dict(),
        os.path.join('models', f'{config.direction}', f'{config.encoder_name}_to_{config.decoder_name}_projection.pth')
    )

    # if config.decoder == 'gpt-2':
    #     decoder.save_pretrained(
    #         os.path.join('models', f'{config.dataset}_{config.direction}', 'decoder', f'{config.decoder_name}.pth')
    #     )
    # else:
    decoder.save_pretrained(
        os.path.join('models', f'{config.direction}', 'decoder', f'{config.decoder_name}')
    )

if __name__ == '__main__':
    np.random.seed(42)

    config = get_config(args.config)

    wandb.login(key=os.getenv('WANDB_TOKEN'))

    now = datetime.now()
    now = now.strftime('%Y%m%d_%H%M%S')
    run = wandb.init(
        config=config,
        project=f'{config.direction}_{config.encoder_name}_{config.decoder_name}',
        name=f'{config.encoder_name}_{config.decoder_name}_{now}'
    )

    # print(args)
    # print(config)

    main(args, config)