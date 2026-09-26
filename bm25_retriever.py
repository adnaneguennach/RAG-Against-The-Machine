from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("resume.txt", "r") as file:
    resume_text = file.read()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=200,
    chunk_overlap=20,
)
raw_chunks = splitter.split_text(resume_text)

documents = [Document(page_content=chunk) for chunk in raw_chunks]

retriever = BM25Retriever.from_documents(documents)
retriever.k = 2 

query_1 = "Where did Adnane work with Docker and AWS?"
print(f"--- Query 1: '{query_1}' ---")
results_1 = retriever.invoke(query_1)
for doc in results_1:
    print(f"Match:\n{doc.page_content}\n")

query_2 = "Has Adnane worked with background task runners?"
print(f"--- Query 2: '{query_2}' ---")
results_2 = retriever.invoke(query_2)
for doc in results_2:
    print(f"Match:\n{doc.page_content}\n")