import os
from pathlib import Path
from typing import Optional
import ir_measures
from ir_measures import *

import loader
from listwise import gen_rerank
from utils.tools import load_runs
home_dir=str(Path.home())

def main(
    model_name_or_path: str,
    run_path: str, 
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    use_logits: Optional[bool] = False,
    use_alpha: Optional[bool] = False,
    variable_passages: Optional[bool] = False,
    **kwargs,
):
    run = load_runs(run_path, topk=100, output_score=True)
    corpus, queries, qrels = loader.load(ir_datasets_name, query_fields, doc_fields)
    run = {k: v for k, v in run.items() if k in qrels}

    reranked_run = gen_rerank(
        model=model_name_or_path,
        run=run, 
        queries=queries, 
        corpus=corpus,
        use_logits=use_logits, # FIRST
        use_alpha=use_alpha, # number
        variable_passages=variable_passages,
        top_k=100,
        window_size=20,
        step_size=10,
        batched=True,
        context_size=4096,
        rerank_type="text", 
    )

    with open(run_path.replace('runs', 'li_reranked_runs'), 'w') as f:
        for qid in reranked_run:
            for i, (docid, score) in enumerate(reranked_run[qid].items()):
                f.write(f"{qid} Q0 {docid} {i+1} {score} li_rerank\n")

    # evaluation
    r1 = ir_measures.calc_aggregate([nDCG@10], qrels, run)
    r2 = ir_measures.calc_aggregate([nDCG@10], qrels, reranked_run)
    return {
        'model_name_or_path': model_name_or_path, 
        'ir_datasets_name': ir_datasets_name,
        'run_path': run_path,
        'original': r1, 
        'reranked': r2
    }

# starting experiments
os.makedirs(f"{home_dir}/APRIL/li_reranked_runs", exist_ok=True)
# model_name_or_path='castorini/rank_zephyr_7b_v1_full'
model_name_or_path='Qwen/Qwen3-8B'
model_name_or_path='Qwen/Qwen2.5-7B-Instruct'

results = {}
for dataset in ['trec-dl-2019', 'trec-dl-2020']:
    run_path = f"{home_dir}/APRIL/runs/run.msmarco-v1-passage.bm25-{dataset}.txt"
    results[dataset] = main(
        model_name_or_path=model_name_or_path,
        run_path=run_path,
        ir_datasets_name=f'msmarco-passage/{dataset}/judged',
        use_logits=False, use_alpha=False,
        variable_passages=False,
    )

print(results)
