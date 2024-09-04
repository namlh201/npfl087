import os
import argparse
import warnings
from types import SimpleNamespace
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size.')
# parser.add_argument('--epochs', default=5, type=int, help='Number of epochs.')
parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory.')
parser.add_argument('--checkpoints_dir', default=None, type=str, help='Pretrained models checkpoint directory.')
parser.add_argument('--debug', default=False, action='store_true', help='Debug.')
# parser.add_argument('--randomized_factor', default=None, type=float, help='Factor of randomized replacing audio with literal transcript.')

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
os.environ['HF_HOME'] = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'

from dotenv import load_dotenv
import numpy as np
import torch
from torch import nn
import torch.utils
import torch.utils.data
from transformers import AutoFeatureExtractor, PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from transformers import logging
from tqdm import tqdm

load_dotenv()
logging.set_verbosity_error()

from block import Encoder, Projection
from data import DataLoader
from utils import get_config, get_dataset, get_decoder, get_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_one(
    encoder: PreTrainedModel,
    projection: nn.Module,
    decoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    audio_feats: torch.Tensor,
    # transcripts: list[str],
    # translations: list[str],
    special_token_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    encoder.eval()
    projection.eval()
    decoder.eval()

    # tokenizer.pad_token_id = tokenizer.eos_token_id

    bos_tok_id, audio_tok_id, transcript_tok_id, _, _ = special_token_ids
    embed = decoder.get_input_embeddings()

    audio_feats = audio_feats.to(device)

    # print(audio_feats.shape)

    audio_hidden_feats = encoder(audio_feats)
    # print(audio_hidden_feats.shape)
    audio_hidden_feats = projection(audio_hidden_feats)
    # print(audio_hidden_feats.shape)

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
    input_feats = input_feats.bfloat16() if config.decoder != 'gpt-2' else input_feats
    input_feats = input_feats.to(device)

    # print(input_feats.shape)

    attention_masks = torch.ones((input_feats.shape[0], input_feats.shape[1]))
    attention_masks = attention_masks.to(device)

    gen_config = GenerationConfig(
        num_beams=2,
        num_return_sequences=1,
        max_length=1024,
        do_sample=True,
        # repetition_penalty=2.5, 
        # length_penalty=1.0, 
        early_stopping=True,
    )

    # mbr_config = MBRConfig(
    #     num_samples=10,
    #     metric="chrf",
    # )

    pred_transcripts_list = decoder.generate(
        inputs_embeds=input_feats,
        attention_mask=attention_masks,
        generation_config=gen_config,
        # num_beams=2,
        # max_length=1024,  
        # repetition_penalty=2.5, 
        # length_penalty=1.0, 
        # early_stopping=True,
        # pad_token_id=tokenizer.pad_token_id
        # mbr_config=mbr_config,
        # tokenizer=tokenizer
    )

    # pred_transcripts = decoder.generate(
    #     inputs_embeds=input_feats,
    #     attention_mask=attention_masks,
    #     max_length=1024,
    #     mbr_config=mbr_config,
    #     tokenizer=tokenizer
    # )

    # print(pred_transcripts.shape)

    transcripts = []
    translations = []

    for pred_transcripts in pred_transcripts_list:
        pred_transcripts = pred_transcripts.squeeze()

        # print('input length =', input_feats.shape[1])
        # print(pred_transcripts, pred_transcripts.shape)
        # print(tokenizer.decode(pred_transcripts[0]))

        try:
            translation_start_idx = pred_transcripts.tolist().index(tokenizer.get_added_vocab()['<|translation|>'])
            # translation_start_idx = 0

            transcript_tokens = pred_transcripts[:translation_start_idx].squeeze()
            translation_tokens = pred_transcripts[translation_start_idx:-1].squeeze()
            # golden_tokens = padded_translations[0]

            # print(translation_tokens)

            transcript = tokenizer.decode(transcript_tokens, skip_special_tokens=True)
            translation = tokenizer.decode(translation_tokens, skip_special_tokens=True)
        except Exception as e:
            transcript = ''
            translation = ''

        transcripts.append(transcript)
        translations.append(translation)
    # translation = tokenizer.decode(translation_tokens, skip_special_tokens=False)
    # golden = translations[0]

    # wer = edit_distance(translation, golden) / len(golden)

    # loss = translation_loss.item()

    return transcripts, translations

def main(args: argparse.Namespace, config: SimpleNamespace):
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")

    tokenizer, special_token_ids = get_tokenizer(config.decoder)
    # print(tokenizer)

    encoder = Encoder().to(device)
    enc_hidden_size = encoder.get_hidden_size()

    # decoder = GPT2Decoder(len(tokenizer)).to(device)
    # decoder = get_decoder(args.decoder, len(tokenizer), init=args.init_lora, mbr_decode=False).to(device)
    decoder = get_decoder(config.decoder, len(tokenizer), init=False).to(device)
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

    test = get_dataset(name=config.dataset, direction=config.direction, subset=config.test_subset, root=args.data_dir)
    test_loader = DataLoader(test, feature_extractor, batch_size=args.batch_size)

    # IGNORE THIS PART FOR NOW
    proj_state_dict = torch.load(
            os.path.join('models', f'{config.dataset}_{config.direction}', f'hubert_to_{config.decoder}_projection.pth'),
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

    if config.decoder == 'gpt-2':
        decoder.load_pretrained(
            os.path.join('models', f'{config.dataset}_{config.direction}', 'decoder', f'{config.decoder}.pth'),
            device=torch.device(device)
        )
    else:
        decoder.load_pretrained(
            os.path.join('models', f'{config.dataset}_{config.direction}', 'decoder', f'{config.decoder}'),
            is_trainable=False
        )

    print(decoder)

    # bleu = BLEU()
    # chrf = CHRF()
    # ter = TER()

    i = 0

    res_dir = f'res/{config.direction}'
    os.makedirs(res_dir, exist_ok=True)

    src_lang = config.direction.split('-')[0]
    tgt_lang = config.direction.split('-')[1]

    src_f = open(os.path.join(res_dir, f'src.{src_lang}'), 'w')
    can_f = open(os.path.join(res_dir, f'can.{tgt_lang}'), 'w')
    hyp_f = open(os.path.join(res_dir, f'hyp.{tgt_lang}'), 'w')
    ref_f = open(os.path.join(res_dir, f'ref.{tgt_lang}'), 'w')

    for audio_feats, transcripts, translations, misc in tqdm(test_loader):
        # if i == 20:
        #     break

        i += 1

        try:
            pred_transcript, candidate = generate_one(
                encoder,
                projection,
                decoder,
                tokenizer,
                audio_feats=audio_feats,
                # transcripts=transcripts,
                # translations=translations,
                special_token_ids=(bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id)
            )
        except Exception as e:
            print(e)
            candidate = ['']

        # score = {
        #     'wer': edit_distance(candidate, translations[0]) / len(translations[0]),
        #     'bleu': bleu.corpus_score([candidate], [translations]).score,
        #     'chrf': chrf.corpus_score([candidate], [translations]).score,
        #     'ter': ter.corpus_score([candidate], [translations]).score,
        # }

        # score = {
        #     # 'wer': edit_distance(candidate, transcripts[0]) / len(transcripts[0]),
        #     'bleu': bleu.corpus_score(candidate, [transcripts]).score,
        #     'chrf': chrf.corpus_score(candidate, [transcripts]).score,
        #     'ter': ter.corpus_score(candidate, [transcripts]).score,
        # }

        # print(i)
        # print('score:', score)
        # print('transcript:', transcripts[0])
        # print('predicted transcript:', pred_transcript)
        # print('translation:', candidate)
        # print('golden:', translations[0])
        # print()

        # break

        file_id = f"{misc[0]['talk_id']}-{misc[0]['chunk_id']}"

        # print(f'{file_id}\t{transcripts[0]}', file=src_f)
        for _candidate in candidate:
            print(f'{file_id}\t{_candidate}', file=can_f)
        # print(f'{file_id}\t{translations[0]}', file=ref_f)

    src_f.close()
    can_f.close()
    hyp_f.close()
    ref_f.close()


if __name__ == '__main__':
    config = get_config(args.config)

    main(args, config)