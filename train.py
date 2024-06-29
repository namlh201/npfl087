import os
import argparse
from datetime import datetime
import warnings
from types import SimpleNamespace
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
os.environ['HF_HOME'] = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'

from dotenv import load_dotenv
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

from block import Encoder, Projection
from data import DataLoader
from utils import get_config, get_dataset, get_decoder, get_tokenizer

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
    num_epochs: int,
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    # encoder.train()
    # projection.train()
    # decoder.train()

    # grad_accum_int = 4

    for epoch in range(num_epochs):
        mean_loss = 0.0

        valid_size = len(train_loader)

        # step = 0

        print(f'Epoch #{epoch + 1}:')
        for audio_feats, transcripts, translations in tqdm(train_loader):
            # if i == 100:
            #     break

            try:
                loss = train_step(
                    encoder,
                    projection,
                    decoder,
                    tokenizer,
                    audio_feats=audio_feats,
                    transcripts=transcripts,
                    translations=translations,
                    special_token_ids=special_token_ids
                )

                mean_loss += loss.item()

                # loss = loss / grad_accum_int

                # loss.backward()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # if ((i + 1) % grad_accum_int == 0) or (i + 1 == len(train_loader)):
                #     optimizer.step()
                #     lr_scheduler.step()
                #     optimizer.zero_grad()
            except Exception as e:
                valid_size -= 1

                # print(valid_size)
                # print(transcripts)
                # print(translations)
                # print(e)
                # print()
                continue
                # break

            # i += 1

        mean_loss = mean_loss / valid_size

        wandb.log({
            'epoch': epoch + 1,
            'loss': mean_loss
        })

        print(f'Loss = {mean_loss}')

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
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id = special_token_ids
    embed = decoder.get_input_embeddings()

    audio_feats = audio_feats.to(device)

    # audio_attention_masks = torch.ones_like(audio_feats).to(device)
    audio_hidden_feats = encoder(audio_feats) #, attention_mask=audio_attention_masks)
    audio_hidden_feats = projection(audio_hidden_feats)

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
    embeded_transcripts = embed(padded_transcripts).view((config.batch_size, padded_transcripts.shape[1], -1))

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
    embeded_translations = embed(padded_translations).view((config.batch_size, padded_translations.shape[1], -1))

    embeded_bos_token = embed(bos_tok_id).view((args.batch_size, 1, -1))
    embeded_audio_token = embed(audio_tok_id).view((args.batch_size, 1, -1))
    embeded_transcript_token = embed(transcript_tok_id).view((args.batch_size, 1, -1))
    embeded_translation_token = embed(translation_tok_id).view((args.batch_size, 1, -1))
    embeded_eos_token = embed(eos_tok_id).view((args.batch_size, 1, -1))

    input_feats = torch.cat(
        (
            embeded_bos_token, \
            embeded_audio_token, audio_hidden_feats, \
            embeded_transcript_token, embeded_transcripts, \
            embeded_translation_token, embeded_translations, \
            embeded_eos_token
        ),
        dim=1
    )
    input_feats = input_feats.bfloat16() if args.decoder != 'gpt-2' else input_feats
    input_feats = input_feats.to(device)

    # translation_attention_masks = [
    #     [1] * input_feats.shape[0]
    # ]
    translation_labels = [
        [-100] * (embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + \
        # embeded_transcripts.shape[1] + embeded_translation_token.shape[1]) + \
        transcripts[0] + \
        [translation_tok_id.tolist()[0]] + \
        translations[0] + \
        [eos_tok_id.tolist()[0]]
    ]

    # print(translation_labels)
    # translation_attention_masks = torch.tensor(translation_attention_masks).view((args.batch_size, -1)).to(device)
    translation_labels = torch.tensor(translation_labels).view((args.batch_size, -1)).to(device)

    translation_output = decoder(
        inputs_embeds=input_feats,
        # attention_mask=translation_attention_masks,
        labels=translation_labels
    )

    translation_loss = translation_output.loss

    loss = translation_loss

    # beg = embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]

    # print(loss)
    # print(translation_labels)
    # print(tokenizer.decode(translation_labels.tolist()[0][beg:]))

    return loss

def main(args: argparse.Namespace, config: SimpleNamespace):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")

    tokenizer, special_token_ids = get_tokenizer(config.decoder)
    # print(tokenizer)

    encoder = Encoder().to(device)
    enc_hidden_size = encoder.get_hidden_size()

    decoder = get_decoder(config.decoder, len(tokenizer), init=True).to(device)
    dec_hidden_size = decoder.get_hidden_size()

    projection = Projection(enc_hidden_size, dec_hidden_size).to(device)

    bos_tok_id = special_token_ids['bos_token']
    bos_tok_id = torch.full((config.batch_size, ), bos_tok_id).to(device)

    audio_tok_id = special_token_ids['audio_token']
    audio_tok_id = torch.full((config.batch_size, ), audio_tok_id).to(device)

    transcript_tok_id = special_token_ids['transcript_token']
    transcript_tok_id = torch.full((config.batch_size, ), transcript_tok_id).to(device)

    translation_tok_id = special_token_ids['translation_token']
    translation_tok_id = torch.full((config.batch_size, ), translation_tok_id).to(device)

    eos_tok_id = special_token_ids['eos_token']
    eos_tok_id = torch.full((config.batch_size, ), eos_tok_id).to(device)

    train = get_dataset(name=config.dataset, direction=config.direction, subset=config.train_subset, root=args.data_dir)
    train_loader = DataLoader(train, feature_extractor, batch_size=config.batch_size)

    print(decoder)

    optimizer = torch.optim.AdamW(
        list(projection.parameters()) + list(decoder.parameters()),
        lr=config.lr,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, config.epochs * len(train_loader))

    encoder.train()
    projection.train()
    decoder.train()

    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('models', f'{config.dataset}_{config.direction}'), exist_ok=True)
    os.makedirs(os.path.join('models', f'{config.dataset}_{config.direction}', 'decoder'), exist_ok=True)

    train(
        encoder,
        projection,
        decoder,
        tokenizer,
        optimizer,
        lr_scheduler,
        train_loader=train_loader,
        dev_loader=None,
        num_epochs=config.epochs,
        special_token_ids=(bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id)
    )

    torch.save(
        projection.state_dict(),
        os.path.join('models', f'{config.dataset}_{config.direction}', f'hubert_to_{config.decoder}_projection.pth')
    )

    if config.decoder == 'gpt-2':
        decoder.save_pretrained(
            os.path.join('models', f'{config.dataset}_{config.direction}', 'decoder', f'{config.decoder}.pth')
        )
    else:
        decoder.save_pretrained(
            os.path.join('models', f'{config.dataset}_{config.direction}', 'decoder', f'{config.decoder}')
        )

if __name__ == '__main__':
    config = get_config(args.config)

    wandb.login(key=os.getenv('WANDB_TOKEN'))

    now = datetime.now()
    now = now.strftime('%Y%m%d_%H%M%S')
    run = wandb.init(config=config, project=args.config.split('_')[0], name=f'{config.model}_{now}')

    main(args, config)