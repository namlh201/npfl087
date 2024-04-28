import os
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import torch.utils
import torch.utils.data
from torchaudio.functional import edit_distance

os.environ['HF_HOME'] = os.getcwd() + '/checkpoints'

from sacrebleu.metrics import BLEU, CHRF, TER
import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from transformers import logging
from tqdm import tqdm

logging.set_verbosity_error()

from block import Encoder, GPT2Decoder, get_tokenizer
from data import DataLoader
from utils import get_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.')
parser.add_argument('--dataset', default='MUSTC', type=str, choices=['MUSTC', 'LIBRISPEECH'], required=True, help='Dataset name.')
parser.add_argument('--direction', default='en-cs', type=str, choices=['en-en', 'en-cs'], required=True, help='Translation direction.')
parser.add_argument('--train_subset', default='train', type=str, required=True, help='Train subset.')
parser.add_argument('--dev_subset', default='dev', type=str, required=True, help='Dev subset.')
parser.add_argument('--decoder', default='gpt-2', type=str, choices=['gpt-2', 'gemma'], required=True, help='Decoder name.')
parser.add_argument('--train', default=False, action='store_true', help='Train or Eval.')

args = parser.parse_args()

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
    encoder.train()
    decoder.train()

    for epoch in range(num_epochs):
        mean_loss = 0.0

        print(f'Epoch #{epoch + 1}:')
        for audio_feats, transcripts, translations in tqdm(train_loader):
            loss = train_step(
                encoder,
                decoder,
                tokenizer,
                audio_feats=audio_feats,
                transcripts=transcripts,
                translations=translations,
                special_token_ids=special_token_ids
            )

            mean_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        mean_loss = mean_loss / len(train_loader)

        print(f'Loss = {mean_loss}')

        torch.save(
            encoder.project.state_dict(),
            os.path.join('models', f'{args.dataset}_{args.direction}', f'hubert_to_{args.decoder}_projection_e{epoch + 1}.pth')
        )
        torch.save(
            decoder.state_dict(),
            os.path.join('models', f'{args.dataset}_{args.direction}', f'{args.decoder}_e{epoch + 1}.pth')
        )


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
    embeded_transcripts = embed(padded_transcripts).view((args.batch_size, padded_transcripts.shape[1], -1))

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
    embeded_translations = embed(padded_translations).view((args.batch_size, padded_translations.shape[1], -1))

    embeded_audio_token = embed(audio_tok_id).view((args.batch_size, 1, -1))
    embeded_transcript_token = embed(transcript_tok_id).view((args.batch_size, 1, -1))
    embeded_translation_token = embed(translation_tok_id).view((args.batch_size, 1, -1))
    embeded_eot_token = embed(eot_tok_id).view((args.batch_size, 1, -1))

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
    translation_attention_masks = torch.tensor(translation_attention_masks).view((args.batch_size, -1)).to(device)
    translation_labels = torch.tensor(translation_labels).view((args.batch_size, -1)).to(device)

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
#     encoder.eval()
#     decoder.eval()

#     audio_tok_id, transcript_tok_id, translation_tok_id, eot_tok_id = special_token_ids
#     embed = decoder.get_input_embeddings()

#     audio_feats = audio_feats.to(device)

#     audio_hidden_feats = encoder(audio_feats)

#     encoded_transcripts = list(
#         map(
#             lambda transcript: tokenizer(transcript)['input_ids'],
#             transcripts
#         )
#     )
#     padded_transcripts = nn.utils.rnn.pad_sequence(
#         [torch.tensor(transcript) for transcript in encoded_transcripts],
#         batch_first=True
#     ).to(device)
#     embeded_transcripts = embed(padded_transcripts).view((args.batch_size, padded_transcripts.shape[1], -1))

#     encoded_translations = list(
#         map(
#             lambda translation: tokenizer(translation)['input_ids'],
#             translations
#         )
#     )
#     padded_translations = nn.utils.rnn.pad_sequence(
#         [torch.tensor(translation) for translation in encoded_translations],
#         batch_first=True
#     ).to(device)
#     embeded_translations = embed(padded_translations).view((args.batch_size, padded_translations.shape[1], -1))

#     embeded_audio_token = embed(audio_tok_id).view((args.batch_size, 1, -1))
#     embeded_transcript_token = embed(transcript_tok_id).view((args.batch_size, 1, -1))
#     embeded_translation_token = embed(translation_tok_id).view((args.batch_size, 1, -1))
#     embeded_eot_token = embed(eot_tok_id).view((args.batch_size, 1, -1))

#     input_feats = torch.cat(
#         (
#             embeded_audio_token, audio_hidden_feats, \
#             embeded_transcript_token, embeded_transcripts, \
#             embeded_translation_token, embeded_translations, \
#             embeded_eot_token
#         ),
#         dim=1
#     )
#     input_feats = input_feats.to(device)

#     translation_attention_masks = [
#         [1] * input_feats.shape[0]
#     ]
#     translation_labels = [
#         [-100] * (embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + \
#         # embeded_transcripts.shape[1] + embeded_translation_token.shape[1]) + \
#         encoded_transcripts[0] + \
#         [translation_tok_id] + \
#         encoded_translations[0] + \
#         [eot_tok_id]
#     ]
#     translation_attention_masks = torch.tensor(translation_attention_masks).view((args.batch_size, -1)).to(device)
#     translation_labels = torch.tensor(translation_labels).view((args.batch_size, -1)).to(device)

#     translation_output = decoder(inputs_embeds=input_feats, attention_mask=translation_attention_masks, labels=translation_labels)

#     translation_loss = translation_output.loss

#     translation_tokens = translation_output.logits.detach().argmax(dim=-1)
#     translation_tokens = torch.roll(translation_tokens, 1)

#     translation_start_idx = embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + \
#         embeded_transcript_token.shape[1] + embeded_transcripts.shape[1] + \
#         embeded_translation_token.shape[1]

#     translation_tokens = translation_tokens[:, translation_start_idx:-1].squeeze()
#     golden_tokens = padded_translations[0]

#     # print(translation_tokens, len(translation_tokens))
#     # print(golden_tokens, len(golden_tokens))

#     # assert len(translation_tokens) == len(golden_tokens), f'{len(translation_tokens)}, {len(golden_tokens)}'

#     translation = tokenizer.decode(translation_tokens, skip_special_tokens=True)
#     golden = translations[0]

#     wer = edit_distance(translation, golden) / len(golden)

#     loss = translation_loss.item()

#     return loss, translation, wer


def generate_one(
    encoder: nn.Module,
    decoder: nn.Module,
    tokenizer: AutoTokenizer,
    audio_feats: torch.Tensor,
    # transcripts: list[str],
    # translations: list[str],
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    encoder.eval()
    decoder.eval()

    # tokenizer.pad_token_id = tokenizer.eos_token_id

    audio_tok_id, transcript_tok_id, _, _ = special_token_ids
    embed = decoder.get_input_embeddings()

    audio_feats = audio_feats.to(device)

    audio_hidden_feats = encoder(audio_feats)

    # encoded_transcripts = list(
    #     map(
    #         lambda transcript: tokenizer(transcript)['input_ids'],
    #         transcripts
    #     )
    # )
    # padded_transcripts = nn.utils.rnn.pad_sequence(
    #     [torch.tensor(transcript) for transcript in encoded_transcripts],
    #     batch_first=True
    # ).to(device)
    # embeded_transcripts = embed(padded_transcripts).view((BATCH_SIZE, padded_transcripts.shape[1], -1))

    # encoded_translations = list(
    #     map(
    #         lambda translation: tokenizer(translation)['input_ids'],
    #         translations
    #     )
    # )
    # padded_translations = nn.utils.rnn.pad_sequence(
    #     [torch.tensor(translation) for translation in encoded_translations],
    #     batch_first=True
    # ).to(device)
    # embeded_translations = embed(padded_translations).view((BATCH_SIZE, padded_translations.shape[1], -1))

    embeded_audio_token = embed(audio_tok_id).view((args.batch_size, 1, -1))
    embeded_transcript_token = embed(transcript_tok_id).view((args.batch_size, 1, -1))
    # embeded_translation_token = embed(translation_tok_id).view((BATCH_SIZE, 1, -1))
    # embeded_eot_token = embed(eot_tok_id).view((BATCH_SIZE, 1, -1))

    input_feats = torch.cat(
        (
            embeded_audio_token, audio_hidden_feats, \
            embeded_transcript_token
            # embeded_transcripts, \
            # embeded_translation_token, embeded_translations, \
            # embeded_eot_token
        ),
        dim=1
    )
    input_feats = input_feats.to(device)

    attention_masks = torch.ones((input_feats.shape[0], input_feats.shape[1]))
    attention_masks = attention_masks.to(device)

    pred_transcripts = decoder.generate(
        inputs_embeds=input_feats,
        attention_mask=attention_masks,
        num_beams=2,
        max_length=1024,  
        repetition_penalty=2.5, 
        length_penalty=1.0, 
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

    pred_transcripts = pred_transcripts.squeeze()

    # print('input length =', input_feats.shape[1])
    # print(pred_transcripts, pred_transcripts.shape)
    # print(tokenizer.decode(pred_transcripts[0]))

    translation_start_idx = pred_transcripts.tolist().index(tokenizer.get_added_vocab()['<|translation|>'])

    translation_tokens = pred_transcripts[translation_start_idx:-1].squeeze()
    # golden_tokens = padded_translations[0]

    translation = tokenizer.decode(translation_tokens, skip_special_tokens=True)
    # golden = translations[0]

    # wer = edit_distance(translation, golden) / len(golden)

    # loss = translation_loss.item()

    return translation


def main(args: argparse.Namespace):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")

    tokenizer = get_tokenizer(args.decoder)
    # print(tokenizer)

    decoder = GPT2Decoder(tokenizer).to(device)
    dec_hidden_size = decoder.get_hidden_size()

    encoder = Encoder(dec_hidden_size).to(device)

    audio_tok_id = tokenizer('<|audio|>')['input_ids'][0]
    audio_tok_id = torch.full((args.batch_size, ), audio_tok_id).to(device)

    transcript_tok_id = tokenizer('<|transcript|>')['input_ids'][0]
    transcript_tok_id = torch.full((args.batch_size, ), transcript_tok_id).to(device)

    translation_tok_id = tokenizer('<|translation|>')['input_ids'][0]
    translation_tok_id = torch.full((args.batch_size, ), translation_tok_id).to(device)

    eot_tok_id = tokenizer('<|endoftext|>')['input_ids'][0]
    eot_tok_id = torch.full((args.batch_size, ), eot_tok_id).to(device)

    if args.train:
        librispeech_train = get_dataset(name=args.dataset, direction=args.direction, subset=args.train_subset)
        train_loader = DataLoader(librispeech_train, feature_extractor, batch_size=args.batch_size)

        optimizer = torch.optim.Adam(
            list(encoder.project.parameters()) + list(decoder.parameters()),
            lr=0.001,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 10, args.epochs * len(train_loader))

        encoder.train()
        decoder.train()

        os.makedirs('models', exist_ok=True)
        os.makedirs(os.path.join('models', f'{args.dataset}_{args.direction}'), exist_ok=True)

        train(
            encoder,
            decoder,
            tokenizer,
            optimizer,
            lr_scheduler,
            train_loader=train_loader,
            dev_loader=None,
            num_epochs=args.epochs,
            special_token_ids=(audio_tok_id, transcript_tok_id, translation_tok_id, eot_tok_id)
        )

        torch.save(
            encoder.project.state_dict(),
            os.path.join('models', f'{args.dataset}_{args.direction}', f'hubert_to_{args.decoder}_projection.pth')
        )
        torch.save(
            decoder.state_dict(),
            os.path.join('models', f'{args.dataset}_{args.direction}', f'{args.decoder}.pth')
        )
    else:
        librispeech_dev = get_dataset(name=args.dataset, direction=args.direction, subset=args.dev_subset)
        dev_loader = DataLoader(librispeech_dev, feature_extractor, batch_size=args.batch_size)

        encoder.project.load_state_dict(
            torch.load(
                os.path.join('models', f'{args.dataset}_{args.direction}', f'hubert_to_{args.decoder}_projection.pth'),
                map_location=encoder.device
            )
        )
        decoder.load_state_dict(
            torch.load(
                os.path.join('models', f'{args.dataset}_{args.direction}', f'{args.decoder}.pth'),
                map_location=decoder.device
            )
        )

        bleu = BLEU()
        chrf = CHRF()
        ter = TER()

        for audio_feats, transcripts, translations in tqdm(dev_loader):
            candidate = generate_one(
                encoder,
                decoder,
                tokenizer,
                audio_feats=audio_feats,
                # transcripts=transcripts,
                # translations=translations,
                special_token_ids=(audio_tok_id, transcript_tok_id, translation_tok_id, eot_tok_id)
            )

            # score = {
            #     'wer': edit_distance(candidate, translations[0]) / len(translations[0]),
            #     'bleu': bleu.corpus_score([candidate], [translations]).score,
            #     'chrf': chrf.corpus_score([candidate], [translations]).score,
            #     'ter': ter.corpus_score([candidate], [translations]).score,
            # }

            score = {
                'wer': edit_distance(candidate, transcripts[0]) / len(transcripts[0]),
                'bleu': bleu.corpus_score([candidate], [transcripts]).score,
                'chrf': chrf.corpus_score([candidate], [transcripts]).score,
                'ter': ter.corpus_score([candidate], [transcripts]).score,
            }

            print('score:', score)
            print('transcript:', transcripts[0])
            print('translation:', candidate)
            print('golden:', translations[0])

            # break


if __name__ == '__main__':
    main(args)