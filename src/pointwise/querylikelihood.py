from typing import List
from utils.tools import batch_iterator
# from prompts import PromptMode

def rerank(
    model: str,
    run: dict, 
    queries: dict, 
    corpus: dict,
    batch_size: int = 16,
    **kwargs,
):
    tokenizer = model.model.get_tokenizer()

    # prompt preparation
    template = "Please write a question based on this passage.\nPassage: {doc}\nQuestion: {query}"
    id_pairs, prompts, query_lengths = [], [], []
    print(len(run))
    for qid in run:
        query_length = len(tokenizer(queries[qid], add_special_tokens=False)['input_ids'])
        for docid in run[qid]:
            prompts.append(template.format(
                doc=corpus[docid]["contents"], query=queries[qid]
            ))
            id_pairs.append((qid, docid))
            query_lengths.append(query_length)

    # batch inference
    scores = []
    for start, end in batch_iterator(prompts, size=batch_size, return_index=True):
        batch_prompts = prompts[start:end]
        batch_query_lengths = query_lengths[start:end]

        _, batch_scores = model.generate(
            batch_prompts, max_tokens=1, query_lengths=batch_query_lengths
        )
        scores += batch_scores

    # sorting
    reranked_run = {}
    for i in range(len(scores)):
        qid, docid, _ = id_pairs[i]
        reranked_run[qid][docid] = scores[i]

    return reranked_run
