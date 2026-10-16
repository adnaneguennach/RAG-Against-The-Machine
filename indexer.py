from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pathlib import Path

class RAG_Indexer():
    def __init__(self, path = "data/vllm-0.10.1", max_chunk_size = 2000):
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language = Language.PYTHON,
            chunk_size = max_chunk_size, 
            chunk_overlap = 50, 
            add_start_index = True
        )
        self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language = Language.MARKDOWN,
            chunk_size = max_chunk_size,
            chunk_overlap = 50,
            add_start_index = True
        )
        self.path = Path(path)

    def file_reader(self, path):
        with open(path, "r", encoding = "utf-8") as f:
            return f.read()
    

    #function that gets us all the files, then split them to chunks of documents, contains content:
    #format of Document : Document(metadata={'file_path': "path", start_index = 0}, page_content = "lorem ipsum")
    
    def ingest_corpus(self):
        files = list(self.path.rglob("*.py")) + list(self.path.rglob("*.md")) + list(self.path.rglob("*.txt"))
        master_chunks = []
        #we using append instead of extend, bcs extend take an iterable loops on it and adds, append take it and add it directly 
        # extend = [1,2,3,[4,5]] | append =  [1,2,3,4,5]
        for file in files:
            content = self.file_reader(file)
            if file.suffix == ".py":
                doc = self.python_splitter.create_documents(
                    texts = [content],
                    metadatas = [{"file_path ": str(file)}]
                )
            elif file.suffix in [".md", ".txt"] : 
                doc = self.markdown_splitter.create_documents(
                    texts = [content],
                    metadatas = [{"file_path" : str(file)}]
                )
            master_chunks.append(doc)
        return master_chunks[0]

    

    
obj = RAG_Indexer()
print(obj.ingest_corpus())