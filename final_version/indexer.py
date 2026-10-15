from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from typing import List
from pathlib import Path 
import bm25s
import json
import fire
from pydantic import BaseModel

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class RAG:
    def __init__(self, x):
        self.splitter_py = RecursiveCharacterTextSplitter.from_language(
                language= Language.PYTHON, 
                chunk_size = 2000,
                chunk_overlap = 50,
                add_start_index= True
                )
        self.splitter_txt = RecursiveCharacterTextSplitter.from_language(
            language = Language.MARKDOWN,
            chunk_size = 2000, 
            chunk_overlap = 50,
            add_start_index= True
        )
        self.path = Path(x)
        self.content = ""

    def _retriever(self, path):
        with open(path, "r", encoding= "utf-8") as f:
            return f.read()
    
    def ingest_corpus(self):
        files = list(self.path.rglob("*.py")) + list(self.path.rglob("*.md")) + list(self.path.rglob("*.txt"))
        master_chunk = []
        for file in files:
            content = self._retriever(file)

            if file.suffix == ".py":
                doc = self.splitter_py.create_documents(
                     texts = [content],
                     metadatas = [{"file_path": str(file)}]
                    )            
                # print(doc)
                # exit()
            elif file.suffix in [".md", ".txt"]:
                doc = self.splitter_txt.create_documents(
                     texts = [content],
                     metadatas = [{"file_path": str(file)}]
                    )  
        
            master_chunk.extend(doc)
        return master_chunk
    
    def index_chunks(self):
        master_chunks = self.ingest_corpus()
        min_list  = []
        raw_corpus : list[str] = []

        for chunk in master_chunks:
            start_idx = int(chunk.metadata["start_index"])
            end_idx = start_idx + len(chunk.page_content)
            source_obj = MinimalSource(
                        file_path = chunk.metadata["file_path"],
                        first_character_index = start_idx, 
                        last_character_index = end_idx
                    )
        
            min_list.append(source_obj)
            raw_corpus.append(chunk.page_content)
        
        return min_list, raw_corpus

    def build_and_save_index(
        self,
        sources,
        corpus_texts: List[str],
        output_dir: str = "data/processed",
    ) -> None:
        if len(sources) != len(corpus_texts):
            raise ValueError(
                "Each corpus chunk must have matching source metadata."
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print("Tokenizing corpus chunks...")
        corpus_tokens = bm25s.tokenize(corpus_texts)

        print("Building BM25 index...")
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens)

        bm25.save(str(out_path))
        print(f"BM25 index saved to {out_path}")

        metadata_file = out_path / "metadata.json"
        metadata_payload = []

        for source, text in zip(sources, corpus_texts):
            metadata = source.model_dump()
            metadata["text"] = text
            metadata_payload.append(metadata)

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

        print(f"Metadata saved to {metadata_file}")

# split text 
# bm25 to get best chunk to work with     

if __name__ == '__main__':
    x = RAG("test")
    # fire.Fire(x)    
    min_sources, raw_corpus = x.index_chunks()
    x.build_and_save_index(min_sources,raw_corpus)
    # print(x.ingest_corpus())

    
            


