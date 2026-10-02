from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

vllm_path = Path("vllm-0.10.1")
rec_load = list(vllm_path.rglob("*.py"))

chunked_data = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True
)

def retrieve_data(path):
    with open(path, "r") as f:
        content = f.read()
    return content

i = 0

while (i < len(rec_load)):
    data = retrieve_data(rec_load[i])
    print(chunked_data.split_text(data))
    i += 1

