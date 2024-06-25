import os
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
parser.add_argument('--epochs', default=5, type=int, help='Number of epochs.')
parser.add_argument('--dataset', default='MUSTC', type=str, choices=['MUSTC', 'LIBRISPEECH'], required=True, help='Dataset name.')
parser.add_argument('--direction', default='en-cs', type=str, choices=['en-en', 'en-cs'], required=True, help='Translation direction.')
parser.add_argument('--train_subset', default='train', type=str, required=True, help='Train subset.')
parser.add_argument('--dev_subset', default='dev', type=str, required=True, help='Dev subset.')
parser.add_argument('--decoder', default='gpt-2', type=str, choices=['gpt-2', 'gemma'], required=True, help='Decoder name.')
parser.add_argument('--init_lora', default=False, action='store_true', help='Whether init with LoRA or not.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')
parser.add_argument('--train', default=False, action='store_true', help='Train or Eval.')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'

os.environ['HF_HOME'] = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from sacrebleu.metrics import BLEU, CHRF, TER
import torch
from torch import nn
import torch.utils
import torch.utils.data
from torchaudio.functional import edit_distance
from transformers import AutoFeatureExtractor, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from transformers import logging
from tqdm import tqdm

logging.set_verbosity_error()

from block import Encoder, Projection
from data import DataLoader
from utils import get_dataset, get_decoder, get_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(
    encoder: nn.Module,
    projection: nn.Module,
    decoder: nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    dev_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    # encoder.train()
    # projection.train()
    # decoder.train()

    # grad_accum_int = 4

    for epoch in range(num_epochs):
        mean_loss = 0.0

        valid_size = len(train_loader)

        # i = 0

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

                print(valid_size)
                print(transcripts)
                print(translations)
                print(e)
                print()
                continue
                # break

            # i += 1

        mean_loss = mean_loss / valid_size

        print(f'Loss = {mean_loss}')

        os.makedirs(os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}'), exist_ok=True)
        os.makedirs(os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder'), exist_ok=True)

        torch.save(
            projection.state_dict(),
            os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', f'hubert_to_{args.decoder}_projection_e{epoch + 1}.pth')
        )

        if args.decoder == 'gpt-2':
            decoder.save_pretrained(os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder', f'{args.decoder}.pth'))
        elif args.decoder == 'gemma':
            decoder.save_pretrained(os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder'))

        # decoder.save_pretrained(
        #     os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder', f'{args.decoder}_e{epoch + 1}.pth')
        # )
        # torch.save(
        #     decoder.state_dict(),
        #     os.path.join('models', f'{args.dataset}_{args.direction}', f'e{epoch + 1}', 'decoder', f'{args.decoder}_e{epoch + 1}.pth')
        # )


def train_step(
    encoder: nn.Module,
    projection: nn.Module,
    decoder: nn.Module,
    tokenizer: AutoTokenizer,
    audio_feats: torch.Tensor,
    transcripts: list[str],
    translations: list[str],
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id = special_token_ids
    embed = decoder.get_input_embeddings()

    audio_feats = audio_feats.to(device)

    audio_attention_masks = torch.ones_like(audio_feats).to(device)
    audio_hidden_feats = encoder(audio_feats, attention_mask=audio_attention_masks)
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
    input_feats = input_feats.bfloat16() if args.decoder == 'gemma' else input_feats
    input_feats = input_feats.to(device)

    translation_attention_masks = [
        [1] * input_feats.shape[0]
    ]
    translation_labels = [
        [-100] * (embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]) + \
        # embeded_transcripts.shape[1] + embeded_translation_token.shape[1]) + \
        transcripts[0] + \
        [translation_tok_id.tolist()[0]] + \
        translations[0] + \
        [eos_tok_id.tolist()[0]]
    ]

    # print(translation_labels)
    translation_attention_masks = torch.tensor(translation_attention_masks).view((args.batch_size, -1)).to(device)
    translation_labels = torch.tensor(translation_labels).view((args.batch_size, -1)).to(device)

    translation_output = decoder(inputs_embeds=input_feats, attention_mask=translation_attention_masks, labels=translation_labels)

    translation_loss = translation_output.loss

    loss = translation_loss

    # beg = embeded_bos_token.shape[1] + embeded_audio_token.shape[1] + audio_hidden_feats.shape[1] + embeded_transcript_token.shape[1]

    # print(loss)
    # print(translation_labels)
    # print(tokenizer.decode(translation_labels.tolist()[0][beg:]))

    return loss


def generate_one(
    encoder: nn.Module,
    projection: nn.Module,
    decoder: nn.Module,
    tokenizer: AutoTokenizer,
    audio_feats: torch.Tensor,
    # transcripts: list[str],
    # translations: list[str],
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    encoder.eval()
    projection.eval()
    decoder.eval()

    # tokenizer.pad_token_id = tokenizer.eos_token_id

    bos_tok_id, audio_tok_id, transcript_tok_id, _, _ = special_token_ids
    embed = decoder.get_input_embeddings()

    audio_feats = audio_feats.to(device)

    print(audio_feats.shape)

    audio_hidden_feats = encoder(audio_feats)
    print(audio_hidden_feats.shape)
    audio_hidden_feats = projection(audio_hidden_feats)
    print(audio_hidden_feats.shape)

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

    embeded_bos_token = embed(bos_tok_id).view((args.batch_size, 1, -1))
    embeded_audio_token = embed(audio_tok_id).view((args.batch_size, 1, -1))
    embeded_transcript_token = embed(transcript_tok_id).view((args.batch_size, 1, -1))
    # embeded_translation_token = embed(translation_tok_id).view((BATCH_SIZE, 1, -1))
    # embeded_eot_token = embed(eot_tok_id).view((BATCH_SIZE, 1, -1))

    input_feats = torch.cat(
        (
            embeded_bos_token, \
            embeded_audio_token, audio_hidden_feats, \
            embeded_transcript_token
            # embeded_transcripts, \
            # embeded_translation_token, embeded_translations, \
            # embeded_eot_token
        ),
        dim=1
    )
    input_feats = input_feats.bfloat16() if args.decoder == 'gemma' else input_feats
    input_feats = input_feats.to(device)

    print(input_feats.shape)

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
        # pad_token_id=tokenizer.pad_token_id
    )

    print(pred_transcripts.shape)

    pred_transcripts = pred_transcripts.squeeze()

    # print('input length =', input_feats.shape[1])
    # print(pred_transcripts, pred_transcripts.shape)
    # print(tokenizer.decode(pred_transcripts[0]))

    # translation_start_idx = pred_transcripts.tolist().index(tokenizer.get_added_vocab()['<|translation|>'])
    translation_start_idx = 0

    translation_tokens = pred_transcripts[translation_start_idx:].squeeze()
    # golden_tokens = padded_translations[0]

    # print(translation_tokens)

    # translation = tokenizer.decode(translation_tokens, skip_special_tokens=True)
    translation = tokenizer.decode(translation_tokens, skip_special_tokens=False)
    # golden = translations[0]

    # wer = edit_distance(translation, golden) / len(golden)

    # loss = translation_loss.item()

    return translation


def main(args: argparse.Namespace):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")

    tokenizer, special_token_ids = get_tokenizer(args.decoder)
    # print(tokenizer)

    encoder = Encoder().to(device)
    enc_hidden_size = encoder.get_hidden_size()

    # decoder = GPT2Decoder(len(tokenizer)).to(device)
    decoder = get_decoder(args.decoder, len(tokenizer), init=args.init_lora).to(device)
    dec_hidden_size = decoder.get_hidden_size()

    projection = Projection(enc_hidden_size, dec_hidden_size).to(device)

    bos_tok_id = special_token_ids['bos_token']
    bos_tok_id = torch.full((args.batch_size, ), bos_tok_id).to(device)

    audio_tok_id = special_token_ids['audio_token']
    audio_tok_id = torch.full((args.batch_size, ), audio_tok_id).to(device)

    transcript_tok_id = special_token_ids['transcript_token']
    transcript_tok_id = torch.full((args.batch_size, ), transcript_tok_id).to(device)

    translation_tok_id = special_token_ids['translation_token']
    translation_tok_id = torch.full((args.batch_size, ), translation_tok_id).to(device)

    eos_tok_id = special_token_ids['eos_token']
    eos_tok_id = torch.full((args.batch_size, ), eos_tok_id).to(device)

    if args.train:
        librispeech_train = get_dataset(name=args.dataset, direction=args.direction, subset=args.train_subset, root=args.data_dir)
        train_loader = DataLoader(librispeech_train, feature_extractor, batch_size=args.batch_size)

        print(decoder)

        optimizer = torch.optim.AdamW(
            list(projection.parameters()) + list(decoder.parameters()),
            lr=1e-4,
        )
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 10, args.epochs * len(train_loader))

        encoder.train()
        projection.train()
        decoder.train()

        os.makedirs('models', exist_ok=True)
        os.makedirs(os.path.join('models', f'{args.dataset}_{args.direction}'), exist_ok=True)
        os.makedirs(os.path.join('models', f'{args.dataset}_{args.direction}', 'decoder'), exist_ok=True)

        train(
            encoder,
            projection,
            decoder,
            tokenizer,
            optimizer,
            lr_scheduler,
            train_loader=train_loader,
            dev_loader=None,
            num_epochs=args.epochs,
            special_token_ids=(bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id)
        )

        torch.save(
            projection.state_dict(),
            os.path.join('models', f'{args.dataset}_{args.direction}', f'hubert_to_{args.decoder}_projection.pth')
        )

        if args.decoder == 'gpt-2':
            decoder.save_pretrained(
                os.path.join('models', f'{args.dataset}_{args.direction}', 'decoder', f'{args.decoder}.pth')
            )
        elif args.decoder == 'gemma':
            decoder.save_pretrained(
                os.path.join('models', f'{args.dataset}_{args.direction}', 'decoder', f'{args.decoder}')
            )
        
        # torch.save(
        #     decoder.state_dict(),
        #     os.path.join('models', f'{args.dataset}_{args.direction}', 'decoder', f'{args.decoder}.pth')
        # )
    else:
        librispeech_dev = get_dataset(name=args.dataset, direction=args.direction, subset=args.dev_subset, root=args.data_dir)
        dev_loader = DataLoader(librispeech_dev, feature_extractor, batch_size=args.batch_size)

        # IGNORE THIS PART FOR NOW
        proj_state_dict = torch.load(
                os.path.join('models', f'{args.dataset}_{args.direction}', f'hubert_to_{args.decoder}_projection.pth'),
                map_location=device
            )
        
        if 'weight' in proj_state_dict and 'bias' in proj_state_dict:
            proj_state_dict['project.weight'] = proj_state_dict.pop('weight')
            proj_state_dict['project.bias'] = proj_state_dict.pop('bias')
        # print(proj_state_dict)
        # projection.load_state_dict(
        #     torch.load(
        #         os.path.join('models', f'{args.dataset}_{args.direction}', f'hubert_to_{args.decoder}_projection.pth'),
        #         map_location=projection.device
        #     ),
        # )
        projection.load_state_dict(proj_state_dict)
        # print(projection.state_dict())
        # decoder.load_state_dict(
        #     torch.load(
        #         os.path.join('models', f'{args.dataset}_{args.direction}', 'decoder', f'{args.decoder}.pth'),
        #         map_location=device
        #     )
        # )

        if args.decoder == 'gpt-2':
            decoder.load_pretrained(
                os.path.join('models', f'{args.dataset}_{args.direction}', 'decoder', f'{args.decoder}.pth'),
                device=torch.device(device)
            )
        elif args.decoder == 'gemma':
            decoder.load_pretrained(
                os.path.join('models', f'{args.dataset}_{args.direction}', 'decoder', f'{args.decoder}'),
                is_trainable=False
            )

        print(decoder)

        bleu = BLEU()
        chrf = CHRF()
        ter = TER()

        i = 0

        for audio_feats, transcripts, translations in tqdm(dev_loader):
            if i == 10:
                break

            i += 1

            candidate = generate_one(
                encoder,
                projection,
                decoder,
                tokenizer,
                audio_feats=audio_feats,
                # transcripts=transcripts,
                # translations=translations,
                special_token_ids=(bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id)
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

            print(i)
            print('score:', score)
            print('transcript:', transcripts[0])
            print('translation:', candidate)
            print('golden:', translations[0])
            print()

            # break


if __name__ == '__main__':
    main(args)