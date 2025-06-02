import torch
import math
import vllm
from typing import List

# true_list = [' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
# false_list = [' No', 'No', ' no', 'no', 'NO', ' NO']
# self.yes_tokens = [tokenizer.encode(item, add_special_tokens=False)[0] for item in true_list]
# self.no_tokens = [tokenizer.encode(item, add_special_tokens=False)[0] for item in false_list]

class LLM:

    def __init__(self, 
        model, 
        temperature=0.0,
        dtype='half', gpu_memory_utilization=0.9, 
        num_gpus=1, 
        max_model_len=13712,
        logprobs=None,
        prompt_logprobs=None
    ):
        self.model = vllm.LLM(
            model, 
            dtype=dtype,
            enforce_eager=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature, 
            skip_special_tokens=False,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            max_tokens=2, min_tokens=2, 
        )

        self.logprobs = logprobs
        self.prompt_logprobs = prompt_logprobs
        self.tokenizer = self.model.get_tokenizer()

    def preprocess(self, prompts):
        return prompts

    @torch.no_grad()
    def inference(
        self, 
        x, 
        yes_tokens=None,
        no_tokens=None
    ):
        if isinstance(x, str):
            x = [x]

        x = self.preprocess(x)

        outputs = self.model.generate(x, self.sampling_params, use_tqdm=False)

        prompt_logprobs = [o.outputs[0].logprobs for o in outputs]
        scores = []
        for p in prompt_logprobs:
            yes_ = math.exp(max( [-1e2]+[p[0][i].logprob for i in yes_tokens if (i in p[0])] ))
            no_ = math.exp(max( [-1e2]+[p[0][i].logprob for i in no_tokens if (i in p[0])] ))
            scores.append( (yes_) / (no_ + yes_) )

        return scores
