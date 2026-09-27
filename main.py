from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.documents import Document
from BM25Retriever.retrievers import BM25Retriever
import sys 

class RAG:
    def __init__(self):
        self.content = ""
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def retrieve_data(self, path):
        with open(path, "r") as f:
            self.content = f.read()
    
    def chunker(self):
        chunked_data = RecursiveCharacterTextSplitter(
            # separators = ["\n\n", "", "\n", " "],
            chunk_size = 200,
            chunk_overlap = 20
        )
        self.retrieve_data("resume.txt")
        return chunked_data.split_text(self.content)
    
    def bm25_handler(self, query):
        data = self.chunker()
        chunked_data = [chunk.lower().split() for chunk in data]
        bm25 = BM25Okapi(chunked_data)

        tokenized_query = query.lower().split()
        top_result = bm25.get_top_n(tokenized_query, data, n=1)

        return top_result[0]
    
    def bm25_lgchain(self, query):
        data = self.chunker()
        # docs = [Document(page_content = chunk for chunk in chunks)]
        retriever = BM25Retriever.from_texts(data, k=1)
        result = retriever.invoke(query)
        return result[0]

    def generate_answer(self, query):
        context = self.bm25_lgchain(query)
        
        prompt = f"""Answer the question based strictly on the provided context.

        Context:
        {context}

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

resume_rag = RAG()
resume_rag.generate_answer(sys.argv[1])