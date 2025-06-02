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
    def inference(self, x, kv_cache=None):
        if isinstance(x, str):
            x = [x]

        x = self.preprocess(x)
        inputs = self.tokenizer(x, padding=True, return_tensors="pt").to(self.model.device)

        if kv_cache is None:
            outputs = self.model.model(**inputs, use_cache=True)
            kv_cache = outputs.past_key_values
            return kv_cache
        else:
            outputs = self.model(input_ids=inputs['input_ids'], use_cache=True, past_key_values=kv_cache, logits_to_keep=1)
            logits = outputs.logits[:, -1:, :]
        return logits.detach().cpu()
