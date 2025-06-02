from time import time
from vllm import LLM, SamplingParams
from utils.tools import batch_iterator
from itertools import combinations

query = "How to improve cardiovascular health?"
passages = [
    "Engaging in at least 150 minutes of moderate aerobic exercise weekly significantly boosts heart health.",
    "Eating a balanced diet rich in fruits, vegetables, and whole grains can improve cardiovascular function.",
    "Regular blood pressure monitoring helps in early detection and prevention of heart disease.",
    "Managing cholesterol levels through lifestyle changes and medication protects arteries from plaque buildup.",
    "Reducing sodium intake can lower blood pressure and reduce heart disease risk.",
    "Getting at least seven hours of quality sleep per night supports overall vascular function.",
    "Reducing alcohol consumption has a positive effect on many aspects of physical wellness.",
    "Spending time outdoors can improve mental health and promote light physical activity.",
    "Learning a musical instrument can help improve cognitive functions in adults.",
    "Visiting art galleries fosters creativity and social engagement."
]

labels = [chr(ord('A') + i) for i in range(10)]  # ['A', 'B', 'C', ..., 'J']
pairs = list(combinations(labels, 2))
suffixes = [f"Compare passage {a} and passage {b}, which is more relevant to the query. Answer only {a} or {b}. Answer:" for a, b in pairs]

def build_listwise_prompt(query, passages, use_alpha=True):
    prompt = f"Query: {query}\nPassages:\n"

    for idx, passage in enumerate(passages):
        identifier = chr(ord('A') + idx) if use_alpha else str(idx)
        prompt += f"{identifier}: {passage.strip()}\n"

    prompt += "\n"  # Add spacing to separate suffix clearly
    return prompt
		
def create_prompt(passages, suffix, shuffle=False):
    if shuffle:
        passages_ = [passages[i] for i in random.sample(range(len(passages)), len(passages))]
    else:
        passages_ = passages

    shared_prefix = build_listwise_prompt(query, passages_)
    return shared_prefix + suffix

# Inference with vLLM (cached)
model = LLM(model="Qwen/Qwen2.5-7B-Instruct", enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=2)

for i in range(len(labels)):
    start = time()
    requests = [create_prompt(passages, suffix) for suffix in suffixes]
    outputs = model.generate(requests, sampling_params)
    outputs = [o.outputs[0].text.replace('\n', '') for o in outputs]
    print(f'entire elapsed {i}:= ', time() - start, list(zip(pairs, outputs)))

# Inference with vLLM (without cached)
# model = LLM(model="Qwen/Qwen2.5-7B-Instruct", enable_prefix_caching=False)
#
# for i in range(10):
#     start = time()
#     requests = [create_prompt(passages, suffix, shuffle=True) for suffix in suffixes]
#     outputs = model.generate(requests)
#     print(f'entire elapsed {i}:= ', time() - start)

# for output in outputs:
#     logprobs = [output.prompt_logprobs[i][id].logprob if i > 0 else -1 for i, id in enumerate(output.prompt_token_ids)][-len(prompt_tokens):]
#     print(tokenizer.tokenize(' This is a testing query.'))
#     print(prompt_tokens)
#     print(logprobs)

# doc = "An AI model is a computer program that uses algorithms and data to perform tasks that typically require human intelligence, such as understanding natural language, recognizing patterns, and making decisions."
# query = "What is an AI model?"
# template = f"Passage: {doc}\nQuery: {query}\nIs this passage relevant to the query? Please answer Yes or No.\nAnswer:"
# from llm.hf_encode import LLM
# model = LLM(model="meta-llama/Llama-3.2-1B-Instruct", model_class='clm', temperature=0)
# model.inference(x=[template, template], max_tokens=1)

# import loader
# from utils.tools import load_runs
# import ir_measures
# from ir_measures import *
# run_path=f"/home/jju/APRIL/runs/run.msmarco-v1-passage.bm25-default.dl19.txt"
# ir_datasets_name='msmarco-passage/trec-dl-2019/judged'
#
# run = load_runs(run_path, topk=100, output_score=True)
# _, _, qrels = loader.load(ir_datasets_name)
# optimal_run = {}
# for qid in run:
#     optimal_run[qid] = {}
#     for i, docid in enumerate(qrels[qid]):
#         if docid in run[qid]:
#             score = qrels[qid][docid] + (1/(i+1))
#         else:
#             score = (1/(i+1))
#         optimal_run[qid].update({docid: score})
#
# print(ir_measures.calc_aggregate([nDCG@10, nDCG@20], qrels, optimal_run))

# import loader
# from listwise import gen_rerank
# from utils.tools import load_runs
# import ir_measures
# from ir_measures import *
# corpus, queries, qrels = loader.load('msmarco-passage/trec-dl-2019/judged')
#
# run = load_runs("/home/jju/APRIL/runs/run.msmarco-v1-passage.bm25-default.dl19.txt", topk=100, output_score=True)
#
# output = gen_rerank(
#     "castorini/first_mistral",
#     run=run, 
#     queries=queries, 
#     corpus=corpus,
#     use_logits=True,
#     use_alpha=True,
#     top_k=100,
#     window_size=20,
#     step_size=10,
#     batched=True,
#     context_size=32768,
#     rerank_type="text", 
# )
# print(ir_measures.calc_aggregate([nDCG@10], qrels, run))
# print(ir_measures.calc_aggregate([nDCG@10], qrels, output))


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# # Load model and tokenizer (example: a small GPT-2 variant)
# model_name = "castorini/first_mistral",
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).eval()
#
# # Assume passages are preloaded
# import loader
# from utils.tools import load_runs
# import ir_measures
# from ir_measures import *
# corpus, queries, qrels = loader.load('msmarco-passage/trec-dl-2019/judged')
#
# run = load_runs("/home/jju/APRIL/runs/run.msmarco-v1-passage.bm25-default.dl19.txt", topk=100, output_score=True)
#
# # only select one query in the run 
# run = {k: v for i, (k, v) in enumerate(run.items()) if i == 0}
#
# ALPH_START_IDX = ord('A') - 1
#
# # Function to build the full prompt
# def add_prefix():
#     system_message = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
#     messages = [{"role": "system", "content": system_message}]
#     return messages
#
# # Inference function
# def rank_pair(passage1_id, passage2_id, max_new_tokens=10):
#
#     # static prompt
#     prompt = build_prompt(passage1_id, passage2_id)
#     dynamic_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
#     # dynamic prompt
#     static_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
#
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             pad_token_id=tokenizer.eos_token_id
#         )
#
#     generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # Extract the very end of the output after the prompt
#     response = generated[len(prompt):].strip()
#     return response
#
# # Example: compare [A] vs [B]
# result = rank_pair("A", "B")
# print(f"More relevant: {result}")
#
# def create_prompt(
#     query: str,
#     result: Result,
#     use_alpha: bool, 
#     rank_start: int,
#     rank_end: int,
# ) -> Tuple[str, int]:
#
#     query = result.query
#     num_passages = len(result.hits)
#     max_length = 300
#     while True:
#         messages = list()
#         if self._system_message and self.system_message_supported:
#             messages.append({"role": "system", "content": self._system_message})
#
#         rank = 0
#         input_context = f"""I will provide you with {num_passages} passages, each indicated by a alphabetical identifier []. Read and memorize all passages carefully. Your will use these passages for multiple comparisons based on their relevance to the search query: {query}\n\n"""
#         for hit in result.hits[rank_start:rank_end]:
#             rank += 1
#             content = hit['content'].replace("Title: Content", "").strip()
#             content = " ".join(content.split()[:max_length])
#             identifier = chr(ALPH_START_IDX + rank) if use_alpha else str(rank)
#             input_context += f"[{identifier}] {content}\n"
#
#         if self._system_message and not self.system_message_supported:
#             messages[0]["content"] = self._system_message + "\n " + messages[0]["content"]
#
#         input_context += f"""\nSearch Query: {query}\nBased on the search query, focus on comparing the passages [identifier_cand1] and [identifier_cand2]. Respond only with the identifier of the passage that is more relevant."""
#         messages.append({"role": "user", "content": input_context})
#
#         prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         prompt = fix_text(prompt)
#
#     return prompt, 0
#
# def create_prompt_batched(
#     self,
#     results: List[Result],
#     use_alpha: bool,
#     rank_start: int,
#     rank_end: int,
#     batch_size: int = 32,
# ) -> List[Tuple[str, int]]:
#     def chunks(lst, n):
#         """Yield successive n-sized chunks from lst."""
#         for i in range(0, len(lst), n):
#             yield lst[i : i + n]
#
#     all_completed_prompts = []
#
#     with ThreadPoolExecutor() as executor:
#         for batch in tqdm(chunks(results, batch_size), desc="Processing batches"):
#             completed_prompts = list(
#                 executor.map(
#                     lambda result: self.create_prompt(result, use_alpha, rank_start, rank_end),
#                     batch,
#                 )
#             )
#             all_completed_prompts.extend(completed_prompts)
#     return all_completed_prompts
