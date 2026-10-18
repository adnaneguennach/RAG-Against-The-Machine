import uuid
from indexer import RAG_Indexer
from pathlib import Path
import json
import bm25s
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
        
        
        for idx, score in zip(docs[0], scores[0]):
            index = int(idx)
            # print(index)
            # exit()

            #ignoring indexes that goes far beyond the len of the chunks
            if index < 0 or index > len(self.chunks):
                continue
            
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
        
        return self.results

    # def search(self, query):
    #     # load_corpus=False because you are using your own metadata.json instead
    #     bm25 = bm25s.BM25.load("data/processed", load_corpus=False)
    #     query_tokens = bm25s.tokenize(query)
    #     docs, scores = bm25.retrieve(query_tokens, k=self.config_k)

    #     with open("data/processed/metadata.json", "r", encoding="utf-8") as f:
    #         self.chunks = json.load(f)

    #     top_results = []
        
    #     # docs[0] is just an array of integers representing the winning positions
    #     for i in range(self.config_k):
    #         idx = int(docs[0][i])
    #         score = float(scores[0][i])
            
    #         # print(docs[0])
    #         # exit()
    #         if score <= 0.0:
    #             break
                
    #         winning_chunk = self.chunks[idx]
    #         top_results.append(winning_chunk)
            
    #     return top_results


x = RAG_search(5)
print(x.search("parser.add_argument("))




# import json
# from pathlib import Path

# import bm25s


# class RAG_search:
#     def __init__(self, k: int = 5, path: str = "data/processed"):
#         if k < 1:
#             raise ValueError("k must be greater than 0")

#         self.config_k = k
#         self.path = Path(path)

#         self.bm25 = bm25s.BM25.load(
#             str(self.path),
#             load_corpus=False,
#         )

#         metadata_path = self.path / "metadata.json"

#         with open(metadata_path, "r", encoding="utf-8") as f:
#             self.chunks = json.load(f)

#     # def search(self, query: str) -> list[dict]:
#     #     if not query.strip():
#     #         return []

#     #     query_tokens = bm25s.tokenize([query])

#     #     docs, scores = self.bm25.retrieve(
#     #         query_tokens,
#     #         k=min(self.config_k, len(self.chunks)),
#     #     )

#     #     top_results = []

#     #     for idx, score in zip(docs[0], scores[0]):
#     #         # print(idx)
#     #         # exit(docs)
#     #         idx = int(idx)
#     #         score = float(score)

#     #         if score <= 0.0:
#     #             continue

#     #         if idx < 0 or idx >= len(self.chunks):
#     #             continue

#     #         chunk = self.chunks[idx]

#     #         file_path = chunk.get(
#     #             "file_path",
#     #             chunk.get("file_path ", ""),
#     #         )

#     #         start_index = chunk.get(
#     #             "first_character_index",
#     #             chunk.get("start_index", 0),
#     #         )

#     #         last_index = chunk.get("last_character_index")

#     #         if last_index is None:
#     #             content = chunk.get("text", chunk.get("content", ""))
#     #             last_index = (
#     #                 int(start_index) + len(content)
#     #                 if content
#     #                 else int(start_index)
#     #             )

#     #         top_results.append({
#     #             "file_path": str(file_path),
#     #             "first_character_index": int(start_index),
#     #             "last_character_index": int(last_index),
#     #         })

       
#     #     return top_results



#     def search(self, query):
        



    


# import fire
# import json
# import uuid
# from pathlib import Path
# from typing import List
# from pydantic import BaseModel, Field
# from tqdm import tqdm

# # --- 1. Pydantic Data Models ---
# class MinimalSource(BaseModel):
#     file_path: str
#     first_character_index: int
#     last_character_index: int

# class UnansweredQuestion(BaseModel):
#     question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     question: str

# class RagDataset(BaseModel):
#     rag_questions: List[UnansweredQuestion]

# class MinimalSearchResults(BaseModel):
#     question_id: str
#     question: str
#     retrieved_sources: List[MinimalSource]

# class StudentSearchResults(BaseModel):
#     search_results: List[MinimalSearchResults]
#     k: int

# # --- 2. The CLI Wrapper ---
# class RAGCLI:
#     def search(self, query: str, k: int = 5):
#         """Return the top-k sources for a single query[cite: 17]."""
#         retriever = RAG_search(k=k)
#         raw_results = retriever.search(query)
        
#         formatted_result = MinimalSearchResults(
#             question_id=str(uuid.uuid4()),
#             question=query,
#             retrieved_sources=raw_results
#         )
        
#         print(formatted_result.model_dump_json(indent=2))

#     def search_dataset(self, dataset_path: str, k: int, save_directory: str):
#         """Run search over a whole dataset and write a StudentSearchResults JSON file[cite: 17]."""
#         dataset_file = Path(dataset_path)
        
#         # Load the input dataset
#         with open(dataset_file, "r", encoding="utf-8") as f:
#             raw_data = json.load(f)
        
#         dataset = RagDataset(**raw_data)
#         retriever = RAG_search(k=k)
        
#         search_results_list = []
        
#         # Process queries with tqdm progress bar[cite: 17]
#         for q in tqdm(dataset.rag_questions, desc="Searching Dataset"):
#             raw_results = retriever.search(q.question)
            
#             search_results_list.append(
#                 MinimalSearchResults(
#                     question_id=q.question_id,
#                     question=q.question,
#                     retrieved_sources=raw_results
#                 )
#             )
            
#         # Wrap all results in the batch output schema[cite: 14, 15]
#         student_results = StudentSearchResults(
#             search_results=search_results_list,
#             k=k
#         )
        
#         # Ensure the output directory exists and save the file[cite: 17]
#         out_dir = Path(save_directory)
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_file = out_dir / dataset_file.name
        
#         with open(out_file, "w", encoding="utf-8") as f:
#             f.write(student_results.model_dump_json(indent=2))
            
#         print(f"Saved student_search_results to {out_file}")

# if __name__ == "__main__":
#     fire.Fire(RAGCLI)