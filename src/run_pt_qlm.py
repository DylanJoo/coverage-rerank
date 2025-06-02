from llm.vllm_back import LLM
import loader
from pointwise import qlm_rerank
from typing import Optional
from utils.tools import load_runs
import logging
logger = logging.getLogger(__name__)

def main(
    model_name_or_path: str,
    run_path: str, 
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
    **kwargs,
):
    run = load_runs(run_path, topk=100)
    corpus, queries, qrels = loader.load(ir_datasets_name, query_fields, doc_fields)
    run = {k: v for k, v in run.items() if k in qrels}

    model = LLM(
        model=model_name_or_path,
        temperature=0.6, top_p=1.0,
        prompt_logprobs=5,
        gpu_memory_utilization=0.9, 
    )

    reranked_run = qlm_rerank(
        model=model,
        run=run,
        queries=queries,
        corpus=corpus,
        batch_size=8
    )

    # evaluation
    result = ir_measures.calc_aggregate([nDCG@10], qrels, reranked_run)[nDCG@10]
    print(result)

main(
    model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
    run_path="/home/jju/APRIL/runs/run.msmarco-v1-passage.bm25-rm3-default.dl19.txt",
    ir_datasets_name='msmarco-passage/trec-dl-2019',
)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name_or_path', type=str)
#     parser.add_argument('--ir_datasets_name', type=str)
#     parser.add_argument('--query_fields', type=str, nargs='+')
#     parser.add_argument('--doc_fields', type=str, nargs='+')
#     parser.add_argument('--max_length', type=int, default=512)
#     args = parser.parse_args()
#
#     main()

# kwargs: 
# ("prompt_mode", PromptMode.MONOT5),
# ("context_size", 512),
# ("device", "cuda"),
# ("batch_size", 64),
# ("context_size", 4096),
# ("prompt_mode", PromptMode.RANK_GPT),
# ("num_few_shot_examples", 0),
# ("device", "cuda"),
# ("num_gpus", 1),
# ("variable_passages", False),
# ("window_size", 20),
# ("system_message", None),
# ("sglang_batched", False),
# ("tensorrt_batched", False),
# ("use_logits", False),
# ("use_alpha", False),
