import os

os.environ['HF_HOME'] = os.getcwd() + '/checkpoints'
os.environ['HF_TOKEN'] = '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'

# import numpy as np
import torch
from torch import nn
# import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, HubertForCTC, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    # device = torch.device(device)

    def __init__(self):
        super().__init__()

        # self.tokenizer = tokenizer

        # self.config = HubertConfig(
        #     vocab_size=len(tokenizer.vocab)
        # )

        # self.encoder = HubertForCTC(self.config)

        # self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        # self.sampling_rate = self.feature_extractor.sampling_rate

        # self.encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960") #.to(self.device)

        # self.encoder = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")

        self.encoder = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

        # self.encoder = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

        for params in self.encoder.parameters():
            params.requires_grad = False

        # self.ctc_head = nn.Linear(self.encoder.config.hidden_size, tokenizer.vocab_size) #.to(self.device)

        # self.project = nn.Linear(self.encoder.config.hidden_size, decoder_hidden_size) #.to(self.device)

    def get_hidden_size(self) -> int:
        return self.encoder.config.hidden_size

    # def ctc_decode(self, ids: torch.Tensor) -> torch.Tensor:
    #     tok_ids, tok_id_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(ids)))

    # @property
    # def feature_extractor(self):
    #     return self.encoder.hubert.feature_extractor
    
    # def forward(self, feats: torch.Tensor, attention_mask: torch.Tensor=None) -> torch.Tensor:
    #     # feats = self.feature_extractor(audio, sampling_rate=self.sampling_rate, return_tensors='pt').input_values

    #     feats = self.encoder(feats, attention_mask=attention_mask).last_hidden_state
    #     # feats = self.encoder(feats, attention_mask=attention_mask, output_hidden_states=True)

    #     # print(feats, feats.shape)

    #     hidden = []

    #     for i in range(0, feats.shape[1], 4):
    #         hidden.append(torch.mean(feats[0][i:i + 4], dim=0))

    #     hidden = torch.stack(hidden).view((feats.shape[0], -1, feats.shape[-1]))

    #     # hidden = feats.hidden_states[-1]

    #     # print(hidden.shape)

    #     # logits = feats.logits

    #     # feats = self.encoder(feats).logits

    #     # tok_ids = torch.argmax(logits, dim=-1)

    #     # print(hidden[0])

    #     # print(tok_ids)

    #     # print(feats, feats.shape)

    #     # ctc_out = self.ctc_head(feats)

    #     # CTC decode
    #     # predicted_ids = torch.argmax(ctc_out, dim=-1)

    #     # group same tokens into non-repeating tokens in CTC style decoding
    #     # processed_tok_ids, tok_id_repetitions = zip(*((tok_id, len(list(group_iter))) for tok_id, group_iter in groupby(tok_ids[0])))

    #     # # # filter self.pad_token which is used as CTC-blank token
    #     # # processed_tok_ids = list(filter(lambda tok_id: tok_id != 0, processed_tok_ids))

    #     # # processed_tok_ids = list(filter(lambda i: processed_tok_ids[i] != self.tokenizer.pad_token_id, range(len(processed_tok_ids))))
    #     # processed_tok_ids = list(filter(lambda i: processed_tok_ids[i] != 0, range(len(processed_tok_ids))))

    #     # # print(feats)

    #     # print(processed_tok_ids)

    #     # # feats = feats[:, :, processed_tok_ids]

    #     # hidden = hidden[:, processed_tok_ids, :]

    #     # # logits = logits[:, processed_tok_ids, :]
    #     # feats = tok_ids[:, processed_tok_ids]

    #     # logits = F.log_softmax(logits.permute(1, 0, 2), dim=-1)

    #     # feats = torch.tensor([processed_tok_ids]).to(self.device)

    #     hidden = self.project(hidden)

    #     # print(processed_tok_ids, feats)

    #     # feats = self.project(feats)

    #     return hidden#, feats

    def forward(self, feats: torch.Tensor, attention_mask: torch.Tensor=None) -> torch.Tensor:
        # feats = self.feature_extractor(audio, sampling_rate=self.sampling_rate, return_tensors='pt').input_values

        # feats = self.encoder(feats, attention_mask=attention_mask).last_hidden_state
        feats = self.encoder(feats, attention_mask=attention_mask, output_hidden_states=True)

        # print(feats, feats.shape)

        hidden = feats.hidden_states[-1]

        # print(hidden.shape)

        logits = feats.logits

        # feats = self.encoder(feats).logits

        # CTC decode
        tok_ids = torch.argmax(logits, dim=-1)

        # group same tokens into non-repeating tokens in CTC style decoding
        token_ids, counts = torch.unique_consecutive(tok_ids[0], return_counts=True)
        token_ids = token_ids.tolist()
        counts = counts.tolist()

        filtered_hidden = []

        # filter self.pad_token which is used as CTC-blank token
        # non_blank_token_ids = []
        idx = 0
        for tok_id, count in zip(token_ids, counts):
            if tok_id != 0:
                # non_blank_token_ids.append(idx)
                filtered_hidden.append(torch.mean(hidden[:, idx:idx + count, :], dim=1))

            idx += count

        # print(tok_ids, non_blank_token_ids, token_ids, counts)

        # hidden = hidden[:, non_blank_token_ids, :]

        hidden = torch.stack(filtered_hidden, dim=1)

        # hidden = self.project(hidden)

        return hidden
    
class Projection(nn.Module):
    # device = torch.device(device)

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super().__init__()

        self.project = nn.Linear(encoder_hidden_size, decoder_hidden_size)

    # def load_state_dict(self, state_dict: os.Mapping[str, dict], strict: bool = True, assign: bool = False):
    #     self.project.load_state_dict(state_dict, strict, assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)

class Decoder(nn.Module):
    def get_hidden_size(self) -> int:
        pass

    def get_input_embeddings(self):
        pass

    def save_pretrained(self, model_dir_path: str):
        pass

    def load_pretrained(self, model_dir_path: str):
        pass

class GPT2Decoder(Decoder):
    # device = torch.device(device)

    def __init__(self, vocab_size: int):
        super().__init__()

        # self.tokenizer = tokenizer

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        # lora_config = LoraConfig(
        #     r=8,
        #     target_modules=["c_attn", "c_proj"],
        #     task_type="CAUSAL_LM",
        # )

        # self.decoder = AutoModelForCausalLM.from_pretrained(
        #     'openai-community/gpt2',
        #     torch_dtype=torch.bfloat16,
        #     quantization_config=bnb_config,
        #     device_map={"":0},
        #     # token=os.environ['HF_TOKEN'],
        # )

        # self.decoder = get_peft_model(self.decoder, lora_config)

        self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2") #.to(self.device)
        self.decoder.resize_token_embeddings(vocab_size)

        # for params in self.decoder.parameters():
        #     params.requires_grad = False

    def get_hidden_size(self) -> int:
        return self.decoder.transformer.config.hidden_size
    
    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def forward(self, **kwargs):
        return self.decoder(**kwargs)
    
    def generate(self, **kwargs):
        return self.decoder.generate(**kwargs)
    
    def save_pretrained(self, model_path: str):
        torch.save(
            self.state_dict(),
            model_path
        )

    def load_pretrained(self, model_path: str, device: torch.device):
        self.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

class GemmaDecoder(Decoder):
    # device = torch.device(device)

    def __init__(self, vocab_size: int, init_lora: bool=True):
        super().__init__()

        # self.tokenizer = tokenizer

        self.vocab_size = vocab_size

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.decoder = AutoModelForCausalLM.from_pretrained(
            'google/gemma-2b',
            torch_dtype=torch.bfloat16,
            quantization_config=self.bnb_config,
            device_map={"":0},
            token=os.environ['HF_TOKEN'],
        )

        if init_lora:
            lora_config = LoraConfig(
                r=8,
                target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )

            self.decoder = get_peft_model(self.decoder, lora_config)

        # self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2") #.to(self.device)
        self.decoder.resize_token_embeddings(self.vocab_size)


        # TODO: freeze Gemma's layers or not?????
        # for params in self.decoder.parameters():
        #     params.requires_grad = False

    def get_hidden_size(self) -> int:
        return self.decoder.get_decoder().config.hidden_size
    
    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def forward(self, **kwargs):
        return self.decoder(**kwargs)
    
    def generate(self, **kwargs):
        return self.decoder.generate(**kwargs)
    
    def save_pretrained(self, model_dir_path: str):
        self.decoder.save_pretrained(model_dir_path)

    def load_pretrained(self, model_dir_path: str, is_trainable: bool=True):
        # self.decoder = AutoModelForCausalLM.from_pretrained(
        #     'google/gemma-2b',
        #     torch_dtype=torch.bfloat16,
        #     quantization_config=self.bnb_config,
        #     device_map={"":0},
        #     token=os.environ['HF_TOKEN'],
        # )

        # self.decoder.resize_token_embeddings(self.vocab_size)

        self.decoder = PeftModel.from_pretrained(self.decoder, model_dir_path, is_trainable=is_trainable)