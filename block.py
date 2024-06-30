import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_HOME'] = os.getcwd() + '/checkpoints'
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# import numpy as np
# from mbr import MBR
import torch
from torch import nn
from transformers import (
    BitsAndBytesConfig,
    GPT2LMHeadModel,
    GemmaForCausalLM,
    Gemma2ForCausalLM,
    HubertForCTC,
    LlamaForCausalLM,
    MistralForCausalLM
)
from peft import LoraConfig, get_peft_model, PeftModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    # device = torch.device(device)

    def __init__(self):
        super().__init__()

        self.encoder = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

        for params in self.encoder.parameters():
            params.requires_grad = False

    def get_hidden_size(self) -> int:
        return self.encoder.config.hidden_size

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
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super().__init__()

        self.project = nn.Linear(encoder_hidden_size, decoder_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)

class Decoder(nn.Module):
    # def __init__(self):
    #     self.decoder = None

    # def forward(self, **kwargs):
    #     return self.decoder(**kwargs)
    
    # def generate(self, **kwargs):
    #     return self.decoder.generate(**kwargs)

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
    def __init__(self, vocab_size: int, model_name: str, init_lora: bool=True):
        super().__init__()

        self.vocab_size = vocab_size

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # if mbr_decode:
        #     self.decoder = MBR(GemmaForCausalLM).from_pretrained(
        #         'google/gemma-2b',
        #         torch_dtype=torch.bfloat16,
        #         quantization_config=self.bnb_config,
        #         device_map={"":0},
        #         token=os.environ['HF_TOKEN'],
        #     )
        # else:
        if '2' in model_name:
            self.decoder = Gemma2ForCausalLM.from_pretrained(
                f'google/{model_name}',
                torch_dtype=torch.bfloat16,
                quantization_config=self.bnb_config,
                device_map={"":0},
                token=os.environ['HF_TOKEN'],
            )
        else:
            self.decoder = GemmaForCausalLM.from_pretrained(
                f'google/{model_name}',
                torch_dtype=torch.bfloat16,
                quantization_config=self.bnb_config,
                device_map={"":0},
                token=os.environ['HF_TOKEN'],
            )

        self.decoder.resize_token_embeddings(self.vocab_size)

        if init_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=[
                    "q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"
                ],
                # modules_to_save=["embed_tokens"],
                task_type="CAUSAL_LM",
            )

            self.decoder = get_peft_model(self.decoder, lora_config)

        # self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2") #.to(self.device)


        # TODO: freeze Gemma's layers or not?????
        # for params in self.decoder.parameters():
        #     params.requires_grad = False

    def get_hidden_size(self) -> int:
        return self.decoder.get_decoder().config.hidden_size
    
    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

        # return self.decoder.base_model.model.model.embed_tokens

    def forward(self, **kwargs):
        return self.decoder(**kwargs)
    
    def generate(self, **kwargs):
        return self.decoder.generate(**kwargs)
    
    def save_pretrained(self, model_dir_path: str):
        self.decoder.save_pretrained(model_dir_path)

    def load_pretrained(self, model_dir_path: str, is_trainable: bool=True):
        self.decoder = PeftModel.from_pretrained(self.decoder, model_dir_path, is_trainable=is_trainable)

class LlamaDecoder(Decoder):
    def __init__(self, vocab_size: int, model_name: str, init_lora: bool=True):
        super().__init__()

        self.vocab_size = vocab_size

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # if mbr_decode:
        #     self.decoder = MBR(GemmaForCausalLM).from_pretrained(
        #         'google/gemma-2b',
        #         torch_dtype=torch.bfloat16,
        #         quantization_config=self.bnb_config,
        #         device_map={"":0},
        #         token=os.environ['HF_TOKEN'],
        #     )
        # else:
        self.decoder = LlamaForCausalLM.from_pretrained(
            f'meta-llama/{model_name}',
            torch_dtype=torch.bfloat16,
            quantization_config=self.bnb_config,
            device_map={"":0},
            token=os.environ['HF_TOKEN'],
        )

        self.decoder.resize_token_embeddings(self.vocab_size)

        if init_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=[
                    "q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"
                ],
                task_type="CAUSAL_LM",
            )

            self.decoder = get_peft_model(self.decoder, lora_config)

        # for params in self.decoder.parameters():
        #     params.requires_grad = False

    def get_hidden_size(self) -> int:
        return self.decoder.get_decoder().config.hidden_size
    
    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

        # return self.decoder.base_model.model.model.embed_tokens

    def forward(self, **kwargs):
        return self.decoder(**kwargs)
    
    def generate(self, **kwargs):
        return self.decoder.generate(**kwargs)

    def save_pretrained(self, model_dir_path: str):
        self.decoder.save_pretrained(model_dir_path)

    def load_pretrained(self, model_dir_path: str, is_trainable: bool=True):
        self.decoder = PeftModel.from_pretrained(self.decoder, model_dir_path, is_trainable=is_trainable)

class MistralDecoder(Decoder):
    def __init__(self, vocab_size: int, model_name: str, init_lora: bool=True):
        super().__init__()

        self.vocab_size = vocab_size

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # if mbr_decode:
        #     self.decoder = MBR(GemmaForCausalLM).from_pretrained(
        #         'google/gemma-2b',
        #         torch_dtype=torch.bfloat16,
        #         quantization_config=self.bnb_config,
        #         device_map={"":0},
        #         token=os.environ['HF_TOKEN'],
        #     )
        # else:
        self.decoder = MistralForCausalLM.from_pretrained(
            f'mistralai/{model_name}',
            torch_dtype=torch.bfloat16,
            quantization_config=self.bnb_config,
            device_map={"":0},
            token=os.environ['HF_TOKEN'],
        )

        self.decoder.resize_token_embeddings(self.vocab_size)

        if init_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=[
                    "q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"
                ],
                task_type="CAUSAL_LM",
            )

            self.decoder = get_peft_model(self.decoder, lora_config)

        # for params in self.decoder.parameters():
        #     params.requires_grad = False

    def get_hidden_size(self) -> int:
        return self.decoder.get_decoder().config.hidden_size
    
    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

        # return self.decoder.base_model.model.model.embed_tokens

    def forward(self, **kwargs):
        return self.decoder(**kwargs)

    def generate(self, **kwargs):
        return self.decoder.generate(**kwargs)

    def save_pretrained(self, model_dir_path: str):
        self.decoder.save_pretrained(model_dir_path)

    def load_pretrained(self, model_dir_path: str, is_trainable: bool=True):
        self.decoder = PeftModel.from_pretrained(self.decoder, model_dir_path, is_trainable=is_trainable)