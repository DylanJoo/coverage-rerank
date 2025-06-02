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
        temperature=0.7, 
        top_p=1.0, 
        flash_attention_2=True,
        device='auto'
    ):
        start_time = time.time()

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
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token # during trainin, no padding is needed

        self.temperature = temperature
        self.top_p = top_p

        logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    def preprocess(self, x):
        return x

    def generate(self, x, min_tokens=0, max_tokens=1024, return_logits=False, **kwargs):
        if isinstance(x, str):
            x = [x]

        x = self.preprocess(x)
        inputs = self.tokenizer(x, padding=True, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature, 
            top_p=self.top_p, 
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            return_dict_in_generate=True, 
            output_logits=True
        )
        
        generation = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        output_logits = torch.stack(outputs.logits, dim=1)
        print(generation)
        # print(output_logits.max(-1).indices)
        # print(self.tokenizer.decode(output_logits.max(-1).indices.flatten()))

        del inputs, outputs
        torch.cuda.empty_cache()

        if return_logits:
            return generation, output_logits
        else:
            return generation

# def postprocess(self, outputs: List, verbose=False):
#     if self.think_activated:
#         for i, o in enumerate(outputs):
#             if '</think>' in o:
#                 o_think, o_response, = o.split('</think>')
#                 if verbose:
#                     print(o_think)
#                 outputs[i] = o_response
#     return outputs

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
