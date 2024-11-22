import os
from types import SimpleNamespace

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
    GemmaForCausalLM,
    Gemma2ForCausalLM,
    HubertForCTC,
    LlamaForCausalLM,
    MistralForCausalLM,
    Qwen2ForCausalLM,
    WhisperModel
)
from peft import LoraConfig, get_peft_model, PeftModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_trainable_parameters(model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class Encoder(nn.Module):
    def get_hidden_size(self) -> int:
        pass

    def forward(self, x):
        pass

class HubertEncoder(Encoder):
    # device = torch.device(device)

    def __init__(self, model_name: str, **kwargs):
        super().__init__()

        self.encoder = HubertForCTC.from_pretrained(model_name, attn_implementation="sdpa", **kwargs)

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

        del feats

        return hidden

class WhisperEncoder(Encoder):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()

        self.encoder = WhisperModel.from_pretrained(model_name, attn_implementation="sdpa", **kwargs)
        self.encoder = self.encoder.get_encoder()

    def get_hidden_size(self) -> int:
        return self.encoder.config.d_model
    
    def forward(self, feats: torch.Tensor, attention_mask: torch.Tensor=None) -> torch.Tensor:
        return self.encoder(feats, attention_mask=attention_mask).last_hidden_state

class LengthAdapter(nn.Module):
    def __init__(self, batch_size: int, kernel_size: int=5):
        super().__init__()

        # self.conv = nn.Conv1d(
        #     in_channels=1500,
        #     out_channels=300,
        #     kernel_size=kernel_size,
        #     stride=kernel_size,
        #     padding=kernel_size // 2
        # )

        self.conv = nn.Conv2d(
            in_channels=batch_size,
            out_channels=batch_size,
            kernel_size=(kernel_size, kernel_size),
            stride=(kernel_size, 1),
            padding=(0, kernel_size // 2)
        )

        # self.conv = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=1,
        #     kernel_size=(kernel_size, kernel_size),
        #     stride=(kernel_size, 1),
        #     padding=(0, kernel_size // 2)
        # )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # print(feats.shape)

        return self.conv(feats)

        # return self.conv(feats.unsqueeze(1)).squeeze(1)

class Projection(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super().__init__()

        self.project = nn.Linear(encoder_hidden_size, decoder_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)

        return self.project(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.common_config = {
            'torch_dtype': torch.bfloat16,
            'quantization_config': self.bnb_config,
            'device_map': {"":0},
            'token': os.environ['HF_TOKEN'],
            'attn_implementation': "sdpa"
        }

        self.decoder = None

    # def forward(self, **kwargs):
    #     return self.decoder(**kwargs)
    
    # def generate(self, **kwargs):
    #     return self.decoder.generate(**kwargs)

    def print_trainable_parameters(self) -> None:
        trainable_params = 0
        all_param = 0
        for _, param in self.decoder.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def get_peft_model(self, lora_params: SimpleNamespace) -> PeftModel:
        lora_config = LoraConfig(
            r=lora_params.r,
            lora_alpha=lora_params.alpha,
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head"
            ],
            # modules_to_save=["embed_tokens"],
            task_type="CAUSAL_LM",
        )

        return get_peft_model(self.decoder, lora_config)

    def get_hidden_size(self) -> int:
        pass

    def get_input_embeddings(self):
        pass

    def save_pretrained(self, model_dir_path: str):
        pass

    def load_pretrained(self, model_dir_path: str):
        pass

class GemmaDecoder(Decoder):
    def __init__(self, vocab_size: int, model_name: str, lora_params: SimpleNamespace=None, debug=False, **kwargs):
        super().__init__(vocab_size)

        # self.vocab_size = vocab_size

        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        # if mbr_decode:
        #     self.decoder = MBR(GemmaForCausalLM).from_pretrained(
        #         'google/gemma-2b',
        #         torch_dtype=torch.bfloat16,
        #         quantization_config=self.bnb_config,
        #         device_map={"":0},
        #         token=os.environ['HF_TOKEN'],
        #     )
        # else:
        if 'gemma-2-' in model_name:
            self.decoder = Gemma2ForCausalLM.from_pretrained(
                model_name,
                # torch_dtype=torch.bfloat16,
                # quantization_config=self.bnb_config,
                # device_map={"":0},
                # token=os.environ['HF_TOKEN'],
                **self.common_config,
                **kwargs
            )
        else:
            self.decoder = GemmaForCausalLM.from_pretrained(
                model_name,
                # torch_dtype=torch.bfloat16,
                # quantization_config=self.bnb_config,
                # device_map={"":0},
                # token=os.environ['HF_TOKEN'],
                **self.common_config,
                **kwargs
            )

        self.decoder.resize_token_embeddings(self.vocab_size)

        if debug:
            print(f'{model_name} before LoRA:')
            self.print_trainable_parameters()

        if lora_params:
            # lora_config = LoraConfig(
            #     r=lora_params.r,
            #     lora_alpha=lora_params.alpha,
            #     target_modules=[
            #         "q_proj", "o_proj", "k_proj", "v_proj",
            #         "gate_proj", "up_proj", "down_proj",
            #         "embed_tokens", "lm_head"
            #     ],
            #     # modules_to_save=["embed_tokens"],
            #     task_type="CAUSAL_LM",
            # )

            # self.decoder = get_peft_model(self.decoder, lora_config)

            self.decoder = self.get_peft_model(lora_params)

            if debug:
                print(f'{model_name} after LoRA:')
                self.print_trainable_parameters()

        # self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2") #.to(self.device)

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
    def __init__(self, vocab_size: int, model_name: str, lora_params: SimpleNamespace=None, debug=False, **kwargs):
        super().__init__(vocab_size)

        # self.vocab_size = vocab_size

        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

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
            model_name,
            # torch_dtype=torch.bfloat16,
            # quantization_config=self.bnb_config,
            # device_map={"":0},
            # token=os.environ['HF_TOKEN'],
            **self.common_config,
            **kwargs
        )

        self.decoder.resize_token_embeddings(self.vocab_size)

        if debug:
            print(f'{model_name} before LoRA:')
            self.print_trainable_parameters()

        if lora_params:
            # lora_config = LoraConfig(
            #     r=lora_params.r,
            #     lora_alpha=lora_params.alpha,
            #     target_modules=[
            #         "q_proj", "o_proj", "k_proj", "v_proj",
            #         "gate_proj", "up_proj", "down_proj",
            #         "embed_tokens", "lm_head"
            #     ],
            #     task_type="CAUSAL_LM",
            # )

            self.decoder = self.get_peft_model(lora_params)

            if debug:
                print(f'{model_name} after LoRA:')
                self.print_trainable_parameters()

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
    def __init__(self, vocab_size: int, model_name: str, lora_params: SimpleNamespace=None, debug=False, **kwargs):
        super().__init__(vocab_size)

        # self.vocab_size = vocab_size

        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

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
            model_name,
            # torch_dtype=torch.bfloat16,
            # quantization_config=self.bnb_config,
            # device_map={"":0},
            # token=os.environ['HF_TOKEN'],
            **self.common_config,
            **kwargs
        )

        self.decoder.resize_token_embeddings(self.vocab_size)

        if debug:
            print(f'{model_name} before LoRA:')
            self.print_trainable_parameters()

        if lora_params:
            # lora_config = LoraConfig(
            #     r=lora_params.r,
            #     lora_alpha=lora_params.alpha,
            #     target_modules=[
            #         "q_proj", "o_proj", "k_proj", "v_proj",
            #         "gate_proj", "up_proj", "down_proj",
            #         "embed_tokens", "lm_head"
            #     ],
            #     task_type="CAUSAL_LM",
            # )

            self.decoder = self.get_peft_model(lora_params)

            if debug:
                print(f'{model_name} after LoRA:')
                self.print_trainable_parameters()

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


class QwenDecoder(Decoder):
    def __init__(self, vocab_size: int, model_name: str, lora_params: SimpleNamespace=None, debug=False, **kwargs):
        super().__init__(vocab_size)

        # self.vocab_size = vocab_size

        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        # if mbr_decode:
        #     self.decoder = MBR(GemmaForCausalLM).from_pretrained(
        #         'google/gemma-2b',
        #         torch_dtype=torch.bfloat16,
        #         quantization_config=self.bnb_config,
        #         device_map={"":0},
        #         token=os.environ['HF_TOKEN'],
        #     )
        # else:
        self.decoder = Qwen2ForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.bfloat16,
            # quantization_config=self.bnb_config,
            # device_map={"":0},
            # token=os.environ['HF_TOKEN'],
            **self.common_config,
            **kwargs
        )

        self.decoder.resize_token_embeddings(self.vocab_size)

        if debug:
            print(f'{model_name} before LoRA:')
            self.print_trainable_parameters()

        if lora_params:
            # lora_config = LoraConfig(
            #     r=lora_params.r,
            #     lora_alpha=lora_params.alpha,
            #     target_modules=[
            #         "q_proj", "o_proj", "k_proj", "v_proj",
            #         "gate_proj", "up_proj", "down_proj",
            #         "embed_tokens", "lm_head"
            #     ],
            #     task_type="CAUSAL_LM",
            # )

            self.decoder = self.get_peft_model(lora_params)

            if debug:
                print(f'{model_name} after LoRA:')
                self.print_trainable_parameters()

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