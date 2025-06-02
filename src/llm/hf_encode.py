import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time
import json
import string
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

    # "fid": FiDT5, # unified encoder-decoder
    # "cepe": LlamaForCausalContextLM, # encoder + llama
MODEL_CLASS = {
    "clm": AutoModelForCausalLM
}

class LLM:

    def __init__(self, 
        model,
        model_class='CLM',
        temperature=0.0, 
        top_p=1.0, 
        flash_attention_2=True,
        device='auto',
        **kwargs
    ):

        if flash_attention_2:
            model_kwargs = {
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16
            }
        else:
            model_kwargs = {'torch_dtype': torch.float16}

        self.model = MODEL_CLASS[model_class.lower()].from_pretrained(
            model,
            device_map=device,
            **model_kwargs
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        self.temperature = temperature
        self.top_p = top_p

    def get_tokenizer(self):
        return self.tokenizer

    def preprocess(self, prompts):
        return prompts

    @torch.no_grad()
    def inference(self, x):
        if isinstance(x, str):
            x = [x]

        x = self.preprocess(x)
        inputs = self.tokenizer(x, padding=True, return_tensors="pt").to(self.model.device)

        # outputs = self.model.generate(**inputs, min_new_tokens=0, max_new_tokens=5)
        # generation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        logits = self.model.forward(**inputs).logits
        return logits.detach().cpu()

# class Seq2seqLLM(LLM):
#
#     def load_model(self, model_name_or_path, dtype=torch.float16, flash_attention_2=False):
#
#         if flash_attention_2:
#             model_kwargs = {
#                 'torch_dtype': torch.bfloat16,
#                 'attn_implementation': "flash_attention_2"
#             }
#         else:
#             model_kwargs = {'torch_dtype': dtype}
#
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             model_name_or_path,
#             device_map='auto',
#             **model_kwargs
#         )
#         self.model.eval()
#
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
#
#     def generate(self, x, max_tokens=1024, min_tokens=0, **kwargs):
#
#         if isinstance(x, str):
#             x = [x]
#
#         inputs = self.tokenizer(x, padding=True, return_tensors="pt").to(self.model.device)
#
#         outputs = self.model.generate(
#             **inputs, 
#             min_new_tokens=min_tokens, 
#             max_new_tokens=max_tokens
#         )
#         generation = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         del inputs, outputs
#         torch.cuda.empty_cache()
#         return generation
