class Result:
    def __init__(
        self,
        qid: str,
        query: str,
        hits = None,
        ranking_exec_summary = None,
    ):
        self.qid = qid
        self.query = query
        self.hits = hits
        self.ranking_exec_summary = ranking_exec_summary

    def __repr__(self):
        return str(self.__dict__)

class RankingExecInfo:
    def __init__(
        self, prompt, response: str, input_token_count: int, output_token_count: int
    ):
        self.prompt = prompt
        self.response = response
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count

    def __repr__(self):
        return str(self.__dict__)

