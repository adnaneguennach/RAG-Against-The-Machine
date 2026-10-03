from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pydantic import BaseModel

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class RAGIndexer:

    def __init__(self, raw_data_path: str = "data/raw/vllm-0.10.1", max_chunk_size: int = 2000) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.max_chunk_size = max_chunk_size

        self.py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=100,
            add_start_index=True
        )
        self.md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=100,
            add_start_index=True
        )

    def retrieve_data(self, path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except OSError:
            return ""

    def ingest_corpus(self) -> List:
        files = list(self.raw_data_path.rglob("*.py")) + list(self.raw_data_path.rglob("*.md"))
        master_chunks = []

        for file_path in files:
            content = self.retrieve_data(file_path)
            if not content.strip():
                continue

            splitter = self.py_splitter if file_path.suffix == ".py" else self.md_splitter
            docs = splitter.create_documents(
                texts=[content],
                metadatas=[{"file_path": str(file_path)}]
            )
            master_chunks.extend(docs)

        return master_chunks

    def build_metadata_and_corpus(self) -> Tuple[List[MinimalSource], List[str]]:
        master_chunks = self.ingest_corpus()
        minimal_list: List[MinimalSource] = []
        raw_corpus: List[str] = []

        for chunk in master_chunks:
            start_idx = int(chunk.metadata["start_index"])
            text_len = len(chunk.page_content)

            source_obj = MinimalSource(
                file_path=chunk.metadata["file_path"],
                first_character_index=start_idx,
                last_character_index=start_idx + text_len
            )
            minimal_list.append(source_obj)
            raw_corpus.append(chunk.page_content)

        return minimal_list, raw_corpus

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())


    def build_and_save_index(
        self,
        sources: List[MinimalSource],
        corpus_texts: List[str],
        output_dir: str = "data/processed",
    ) -> None:

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print("Tokenizing corpus chunks...")
        tokenized_corpus = [
            self.tokenize(doc) for doc in tqdm(corpus_texts, desc="Tokenizing", unit="chunk")
        ]

   
        print("Fitting BM25 model...")
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_file = out_path / "bm25.pkl"
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25, f)
        print(f"BM25 index saved to {bm25_file}")


        metadata_file = out_path / "metadata.json"
        metadata_payload = [source.model_dump() for source in sources]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)
        print(f"Metadata saved to {metadata_file}")

indexer = RAGIndexer(raw_data_path="data/raw/vllm-0.10.1", max_chunk_size=2000)
sources, raw_corpus = indexer.build_metadata_and_corpus()

indexer.build_and_save_index(sources, raw_corpus)



