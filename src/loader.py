import csv
import json
import logging
import os
import ir_datasets
from collections import defaultdict
from typing import Optional 
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)

def load(
    ir_datasets_name: str,
    query_fields: Optional[list] = None,
    doc_fields: Optional[list] = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:

    dataset = ir_datasets.load(ir_datasets_name)
    corpus, queries, qrels = {}, {}, {}

    # ids
    query_fields = ['text'] if query_fields is None else query_fields
    doc_fields = ['text'] if doc_fields is None else doc_fields

    logger.info("Loading Corpus...")
    for doc in dataset.docs_iter():
        contents = [getattr(doc, f) for f in doc_fields]
        contents = " ".join(contents)
        corpus[doc.doc_id] = {"contents": contents}
    logger.info("Doc Example: %s", list(corpus.values())[0])

    logger.info("Loading Queries...")
    for query in dataset.queries_iter():
        query_contents = [getattr(query, f) for f in query_fields]
        query_contents = " ".join(query_contents)
        queries[query.query_id] = query_contents
    logger.info("Query Example: %s", list(queries.values())[0])

    logger.info("Loading Qrels...")
    n = 0
    for qrel in dataset.qrels_iter():
        n += 1
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {qrel.doc_id: qrel.relevance}
        else:
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    logger.info("Qrel Example: %s (%s)", n, list(qrels.values())[0])

    return corpus, queries, qrels

