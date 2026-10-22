import uuid
from indexer import RAG_Indexer
from pathlib import Path
import json
import bm25s
from models import StudentSearchResults
from models import MinimalSearchResults, MinimalSource

class RAG_search:
    def __init__(self, k, path = "data/processed"):
        self.indexer = RAG_Indexer(path)
        self.config_k = k
        self.results = []
        self.chunks = ""
        

    def search(self, query):
        bm25 = bm25s.BM25.load("data/processed", load_corpus=False)
        query_tokens = bm25s.tokenize([query])
        docs, scores = bm25.retrieve(query_tokens, k=self.config_k)

        
        with open("data/processed/metadata.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        min_lst = []
        for idx, score in zip(docs[0], scores[0]):
            index = int(idx)

            if index < 0 or index > len(self.chunks):
                continue
            
            chunk = self.chunks[index]
            min_lst.append(chunk)


        MinimalSearchResults(
            question_id = str(uuid.uuid4()),
            question = query,
            retrieved_sources = min_lst
        )
        # self.results = min_obj.model_dump()
        return min_lst

    
    def search_dataset(self, path):
        bm25 = bm25s.BM25.load("data/processed", load_corpus=False)
        
        unanswered_question = Path("datasets_public/public/UnansweredQuestions")

        with open(unanswered_question / "dataset_code_public.json", "r", encoding="utf-8") as f:
            queries = json.load(f)

        with open("data/processed/metadata.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        search_questions = []
        for q in queries["rag_questions"]:
            # search_questions.append(self.search(q["question"], q["question_id"]))
            # self.search()
            # print(q["question_id"])
            search_questions.append(
                {
                    "question_id" : q["question_id"],
                    "question" : q["question"],
                    "retrieved_sources" : self.search(q["question"])
                }
            )


        obj = StudentSearchResults(
            search_results  = search_questions, 
            k = self.config_k
        )
        with open("data/output/test.json", "w") as f:
            json.dump(obj.model_dump(), f, indent = 4)
                
        
 


x = RAG_search(5)
x.search_dataset("data/output/test.json")
# print(x.search("What activation formats does the fused batched MoE layer return in vLLM?", "dsds"))

