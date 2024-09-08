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

args = parser.parse_args()

args.data_dir = args.data_dir if args.data_dir else os.getcwd() + '/data'
os.environ['HF_HOME'] = args.checkpoints_dir if args.checkpoints_dir else os.getcwd() + '/checkpoints'

from dotenv import load_dotenv
# import numpy as np
import torch
# from torch import nn
# import torch.utils
# import torch.utils.data
from transformers import SeamlessM4Tv2ForSpeechToText, SeamlessM4TProcessor
# from transformers import AutoProcessor, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from transformers import logging
from tqdm import tqdm

load_dotenv()
logging.set_verbosity_error()

from data import DataLoader
from utils import get_config, get_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_one(
    seamless_model: SeamlessM4Tv2ForSpeechToText,
    seamless_processor: SeamlessM4TProcessor,
    audio_feats: torch.Tensor
) -> tuple[list[str], list[str]]:
    seamless_model.eval()

    audio_feats = audio_feats.to(device)

    translated_tokens = seamless_model.generate(audio_feats, tgt_lang='deu')

    print(translated_tokens)

    translations = seamless_processor.decode(translated_tokens[0].tolist()[0], skip_special_tokens=False)

    print(translations)

    return [translations]

def main(args: argparse.Namespace, config: SimpleNamespace):
    seamless_model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large").to(device)
    seamless_processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

    # print(seamless_processor, type(seamless_processor).__name__)

    # return

    # tokenizer, special_token_ids = get_tokenizer(config.decoder)
    # print(tokenizer)

    # encoder = Encoder().to(device)
    # enc_hidden_size = encoder.get_hidden_size()

    # # decoder = GPT2Decoder(len(tokenizer)).to(device)
    # # decoder = get_decoder(args.decoder, len(tokenizer), init=args.init_lora, mbr_decode=False).to(device)
    # decoder = get_decoder(config.decoder, len(tokenizer), init=False).to(device)
    # dec_hidden_size = decoder.get_hidden_size()

    # projection = Projection(enc_hidden_size, dec_hidden_size).to(device)

    # bos_tok_id = special_token_ids['bos_token']
    # bos_tok_id = torch.full((args.batch_size, ), bos_tok_id).to(device)

    # audio_tok_id = special_token_ids['audio_token']
    # audio_tok_id = torch.full((args.batch_size, ), audio_tok_id).to(device)

    # transcript_tok_id = special_token_ids['transcript_token']
    # transcript_tok_id = torch.full((args.batch_size, ), transcript_tok_id).to(device)

    # translation_tok_id = special_token_ids['translation_token']
    # translation_tok_id = torch.full((args.batch_size, ), translation_tok_id).to(device)

    # eos_tok_id = special_token_ids['eos_token']
    # eos_tok_id = torch.full((args.batch_size, ), eos_tok_id).to(device)

    test = get_dataset(name=config.dataset, direction=config.direction, subset=config.test_subset, root=args.data_dir)
    test_loader = DataLoader(test, seamless_processor.feature_extractor, batch_size=args.batch_size)

    # # IGNORE THIS PART FOR NOW
    # proj_state_dict = torch.load(
    #         os.path.join('models', f'{config.trained_dataset}_{config.direction}', f'hubert_to_{config.decoder}_projection.pth'),
    #         map_location=device
    #     )
    
    # if 'weight' in proj_state_dict and 'bias' in proj_state_dict:
    #     proj_state_dict['project.weight'] = proj_state_dict.pop('weight')
    #     proj_state_dict['project.bias'] = proj_state_dict.pop('bias')
    # # print(proj_state_dict)
    # # projection.load_state_dict(
    # #     torch.load(
    # #         os.path.join('models', f'{args.dataset}_{args.direction}', f'hubert_to_{args.decoder}_projection.pth'),
    # #         map_location=projection.device
    # #     ),
    # # )
    # projection.load_state_dict(proj_state_dict)
    # print(projection.state_dict())
    # decoder.load_state_dict(
    #     torch.load(
    #         os.path.join('models', f'{args.dataset}_{args.direction}', 'decoder', f'{args.decoder}.pth'),
    #         map_location=device
    #     )
    # )

    # if config.decoder == 'gpt-2':
    #     decoder.load_pretrained(
    #         os.path.join('models', f'{config.trained_dataset}_{config.direction}', 'decoder', f'{config.decoder}.pth'),
    #         device=torch.device(device)
    #     )
    # else:
    #     decoder.load_pretrained(
    #         os.path.join('models', f'{config.trained_dataset}_{config.direction}', 'decoder', f'{config.decoder}'),
    #         is_trainable=False
    #     )

    # print(decoder)

    # bleu = BLEU()
    # chrf = CHRF()
    # ter = TER()

    # i = 0

    res_dir = f'res/{config.direction}'
    os.makedirs(res_dir, exist_ok=True)

    src_lang = config.direction.split('-')[0]
    tgt_lang = config.direction.split('-')[1]

    # asr_f = open(os.path.join(res_dir, f'asr.{src_lang}'), 'w')

    # src_f = open(os.path.join(res_dir, f'src.{src_lang}'), 'w')
    can_f = open(os.path.join(res_dir, f'can.{tgt_lang}'), 'w')
    # hyp_f = open(os.path.join(res_dir, f'hyp.{tgt_lang}'), 'w')
    # ref_f = open(os.path.join(res_dir, f'ref.{tgt_lang}'), 'w')

    for audio_feats, _, _, misc in tqdm(test_loader):
        # if i == 20:
        #     break

        # i += 1

        try:
            candidate = generate_one(
                seamless_model,
                seamless_processor,
                audio_feats=audio_feats,
                # transcripts=transcripts,
                # translations=translations,
                # special_token_ids=(bos_tok_id, audio_tok_id, transcript_tok_id, translation_tok_id, eos_tok_id)
            )
        except Exception as e:
            print(e)
            # pred_transcript = ['']
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

        # for _transcript in pred_transcript:
        #     print(f'{file_id}\t{_transcript}', file=asr_f)

        # print(f'{file_id}\t{transcripts[0]}', file=src_f)
        for _candidate in candidate:
            print(f'{file_id}\t{_candidate}', file=can_f)
        # print(f'{file_id}\t{translations[0]}', file=ref_f)

    # asr_f.close()

    # src_f.close()
    can_f.close()
    # hyp_f.close()
    # ref_f.close()


if __name__ == '__main__':
    config = get_config(args.config)

    main(args, config)