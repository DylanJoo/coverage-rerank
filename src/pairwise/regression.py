import re
import math
from tqdm import tqdm
from typing import List
from utils.tools import batch_iterator
import logging

logger = logging.getLogger(__name__)

system_prompt = """<|im_start|>system\nYou are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query<|im_end|>\n"""
user_prompt = """<|im_start|>user\nI will provide you with {num_passages} passages, each indicated by a alphabetical identifier []. Read and memorize all passages carefully. Your will use these passages for multiple comparisons based on their relevance to the search query: {query}\n\n"""
user_prompt += """"Passages:\n{document_list}\nSearch Query: {query}\nBased on the search query, focus on comparing the passages {cand1} and {cand2}. Respond only with the identifier of the passage that is more relevant. ["""

template = system_prompt + user_prompt # {num_passages} {query} {document_list} {cand1} {cand2} 

def extract_scores(
    batch_logits, 
    yes_tokens, 
    no_tokens
):
    scores = []
    for logits in batch_logits: # (B, L, N)
        yes_ = math.exp(max( [logits[-1, i] for i in yes_tokens] ))
        no_ = math.exp(max( [logits[-1, i] for i in no_tokens] ))
        scores.append( (yes_) / (no_ + yes_) )
    return scores

def rerank(
    model: str,
    run: dict, 
    queries: dict, 
    corpus: dict,
    batch_size: int = 16,
    **kwargs,
):

    # prompt preparation
    id_pairs, prompts = [], []
    for qid in run:
        for docid in run[qid]:
            prompts.append(template.format(doc=corpus[docid]["contents"], query=queries[qid]))
            id_pairs.append((qid, docid))

    # token identifier
    tokenizer = model.tokenizer
    true_list = [' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES']
    false_list = [' No', 'No', ' no', 'no', 'NO', ' NO']
    yes_tokens = [tokenizer.encode(item, add_special_tokens=False)[0] for item in true_list]
    no_tokens = [tokenizer.encode(item, add_special_tokens=False)[0] for item in false_list]

    # batch inference
    logger.info('Number of prompts: {len(prompts)}')
    scores = []
    for start, end in tqdm(
        batch_iterator(prompts, size=batch_size, return_index=True),
        total=len(prompts) // batch_size + 1
    ):
        batch_prompts = prompts[start:end]

        batch_logits = model.inference(batch_prompts)
        batch_scores = extract_scores(batch_logits, yes_tokens, no_tokens)
        scores += batch_scores

    # update scores
    reranked_run = {}
    for i in range(len(scores)):
        qid, docid = id_pairs[i]
        if qid not in reranked_run:
            reranked_run[qid] = {docid: scores[i]}
        else:
            reranked_run[qid][docid] = scores[i]

    # sorting
    sorted_run_dict = {}
    for qid, hit in reranked_run.items():
        sorted_hit = sorted(hit.items(), key=lambda x: x[1], reverse=True) 
        sorted_run_dict[qid] = {docid: rel_score for docid, rel_score in sorted_hit}

    return sorted_run_dict
