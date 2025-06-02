"""
This code is modified from RankFirst repo using rank-llm
https://github.com/gangiswag/llm-reranker/blob/main/scripts/utils/llm_util.py
"""
import csv
import os
import logging
import json
import math
from listwise.rank_llm.rankllm import PromptMode, RankLLM
from listwise.rank_llm.reranker import Reranker
from listwise.rank_llm.rank_listwise_os_llm import RankListwiseOSLLM
from listwise.rank_llm.utils import Result

def convert_run_to_result(
    run,
    queries=None,
    corpus=None
):
    """
    run: {
        "<qid1>": {"<docid>": score, "<docid2>": score, ...}, 
        "<qid2>": ...
    }
    results: [
        Result{query=<query1>, hits=[{"docid": <docid>, "score": <score>, "content": <content>},...], 
        Result{query=<query2>, hits=[...]}
    ]
    """
    results = []
    for qid, hits in run.items():
        query = queries[qid]
        pairs = []
        for docid, score in hits.items():
            pairs.append({'docid': docid, 'score': float(score), 'content': corpus[docid]['contents']})
        results.append(Result(qid=qid, query=query, hits=pairs))
    return results

def rerank(
    model, 
    run, queries, corpus,
    use_logits, use_alpha, 
    variable_passages,
    top_k, 
    window_size, 
    step_size, 
    batched, 
    context_size, 
    rerank_type="text", 
    code_prompt_type="docstring"
):
    system_message = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"

    results_for_rerank = convert_run_to_result(run, queries, corpus)

    # Initialize the ranking model
    agent = RankListwiseOSLLM(
        model=model,
        context_size=context_size,
        prompt_mode=PromptMode.RANK_GPT,
        num_few_shot_examples=0,
        device="cuda",
        num_gpus=1,
        variable_passages=variable_passages,
        window_size=window_size,
        system_message=system_message,
        batched=batched,
        rerank_type=rerank_type,
        code_prompt_type=code_prompt_type
    )

    # Perform reranking
    reranker = Reranker(agent=agent)
    reranked_results = reranker.rerank(
        retrieved_results=results_for_rerank,
        use_logits=use_logits,
        use_alpha=use_alpha,
        rank_start=0,
        rank_end=top_k,
        window_size=window_size,
        step=step_size,
        logging=False,
        batched=batched
    )

    reranked_run = {}
    for result in reranked_results:
        reranked_run[result.qid] = {}
        for rank, hit in enumerate(result.hits, start=1):
            hit['rank'] = rank
            reranked_run[result.qid].update({ hit['docid']: hit['score'] })

    return reranked_run
