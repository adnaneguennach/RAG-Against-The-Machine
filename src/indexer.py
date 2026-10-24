from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pathlib import Path
from .models import MinimalSource
import bm25s
import json 
from tqdm import tqdm

class RAG_Indexer:
    def __init__(self, path = "data/raw/vllm-0.10.1", max_chunk_size = 2000):
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language = Language.PYTHON,
            chunk_size = max_chunk_size, 
            chunk_overlap = 100, 
            add_start_index = True
        )
        self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language = Language.MARKDOWN,
            chunk_size = max_chunk_size,
            chunk_overlap = 100,
            add_start_index = True
        )
        self.path = Path(path)

    def file_reader(self, path):
        try:
            with open(path, "r", encoding = "utf-8") as f:
                return f.read()
        except BaseException as e:
            print("[INDEXER_ERROR]: {e}!")
            raise
    

    #function that gets us all the files, then split them to chunks of documents, contains content:
    #format of Document : Document(metadata={'file_path': "path", start_index = 0}, page_content = "lorem ipsum")

    def ingest_corpus(self):
        try:
            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)
            files = list(self.path.rglob("*.py")) + list(self.path.rglob("*.md")) + list(self.path.rglob("*.txt"))
            master_chunks = []
            for file in files:
                content = self.file_reader(file)
                if file.suffix == ".py":
                    doc = self.python_splitter.create_documents(
                        texts = [content],
                        metadatas = [{"file_path": str(file)}]
                    )
                elif file.suffix in [".md", ".txt"] : 
                    doc = self.markdown_splitter.create_documents(
                        texts = [content],
                        metadatas = [{"file_path" : str(file)}]
                    )
                master_chunks.extend(doc)
            return master_chunks

        except BaseException as e:
            print("Error while ingesting the corpus {e}!")
            raise
  
    def index_corpus(self):
        raw_content = []
        min_list = []
        
        for chunk in self.ingest_corpus():
            start_idx = chunk.metadata["start_index"]
            last_idx = start_idx + len(chunk.page_content)

            min_obj = MinimalSource(
                file_path = chunk.metadata["file_path"],
                first_character_index = start_idx,
                last_character_index = last_idx
            )

            raw_content.append(chunk.page_content)
            min_list.append(min_obj)
        if len(raw_content) == 0 or len(min_list) == 0:
            print("[INDEXER_ERROR]: Nothing found !")
            exit()
        return raw_content, min_list

    def build_save_idx(self, source, min_list, output_dir = "data/processed"):
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                self.path.mkdir(parents=True, exist_ok=True)
            corpus_tokens = bm25s.tokenize(source)
            retriever = bm25s.BM25(corpus = source)
            retriever.index(corpus_tokens)
            retriever.save(str(output_path))

            metadata_file = output_path / "metadata.json"
            metadata_payload = []

            for s in tqdm(min_list, desc="Indexing "):
                metadata_payload.append(s.model_dump())

            with open(metadata_file, "w") as f:
                json.dump(metadata_payload, f, indent=2, ensure_ascii=False)
        except BaseException as e:
            print(f"[INDEXER_ERROR]:{e}")
            exit()
