from dev_hf import APRIL

model_name_or_path='meta-llama/Llama-3.2-1B-Instruct'
model = APRIL(
    model_name=model_name_or_path, 
    device="cpu",
    system_message="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
)

# unit test
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

# labels = [chr(ord('A') + i) for i in range(10)]  # ['A', 'B', 'C', ..., 'J']
# pairs = list(combinations(labels, 2))
# suffixes = [f"Compare passage {a} and passage {b}, which is more relevant to the query. Answer only {a} or {b}. Answer:" for a, b in pairs]

from pairwise.rank_llm.utils import Result
pairs = []
for i, passage in enumerate(passages):
    pairs.append({'docid': f"docid_{i}", 'score': float(1/ (i+1)), 'content': passage})

results = []
results.append(Result(qid='qid_0', query=query, hits=pairs))

model.sliding_windows_batched(
    retrieved_results=results,
    use_logits=True,
    rank_start=0,
    rank_end=10,
    window_size=4,
    step=2
)
