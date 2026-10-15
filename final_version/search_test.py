import json
from pathlib import Path
from pydantic_class import StudentSearchResults, MinimalSearchResults, MinimalSource
import bm25s
# from transformers import AutoModelForCausalLM, AutoTokenizer

import uuid

class BM25Search:
    def __init__(self, index_dir: str = "data/processed"):
        self.chunks = ""
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.results = []


    def _load_index(self, path):
        with open(path, "r") as f:
            self.chunks = json.load(f)

    def search(self, query: str, k: int = 1):
        index_dir = Path("data/processed")

        bm25 = bm25s.BM25.load(
            str(index_dir),
            load_corpus=True,
        )

        query_tokens = bm25s.tokenize([query])
        results, scores = bm25.retrieve(query_tokens, k=k)
        with open(index_dir / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        # matches = []
        self._load_index("data/processed/metadata.json")
        for idx in results[0]:
            index = int(idx)
            chunk = self.chunks[index]
            min_obj = MinimalSearchResults(
                question_id = str(uuid.uuid4()),
                question = query,
                retrieved_sources = [MinimalSource(
                    file_path = chunk["file_path"], 
                    first_character_index = chunk["first_character_index"], 
                    last_character_index = chunk["last_character_index"]
                )]
            )
            self.results.append(min_obj)         
            print(self.results)
            exit()

    def llm_answer(self, query):
       
        self.search(query)

        if len(self.results) < 0:
            return "hamza khsso lfloss"

        prompt = f"""Answer the question based strictly on the provided context.

        Context:
        {self.results[0]}

        Question:
        {query}

        Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        final_answer = full_response.split("Answer:")[-1].strip()
        
        print(f"--- RAG PIPELINE COMPLETE ---")
        print(f"User Query: {query}")
        print(f"BM25 Found: {context}")
        print(f"LLM Says:   {final_answer}")
        
        return final_answer


searcher = BM25Search("data/processed")

searcher.llm_answer(
    "How does the refund policy work?"
)

