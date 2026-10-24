import fire
from .search import RAG_search
from .indexer import RAG_Indexer
from .answer import RAG_answer


class RAG:
    def __init__(self):
        pass
    
    #search 
    def index(self, max_chunk_size):
        index_obj = RAG_Indexer(max_chunk_size = max_chunk_size)
        raw_content, min_list = index_obj.index_corpus()
        index_obj.build_save_idx(raw_content, min_list)

    def search(self, query, k):
        if k == 0:
            return "K must be above 0"
        search_obj = RAG_search(k = k)
        return search_obj.search(query = query)

    def search_dataset(self,dataset_path, k, save_directory):
        if k == 0:
            return "K must be above 0"
        search_obj = RAG_search(k = k)
        search_obj.search_dataset(dataset_path, save_directory)

    def answer(self, query, k):
        if k == 0:
            return "K must be above 0"
        answer_obj = RAG_answer()
        return answer_obj.answer_query(query, k)
    

    def answer_dataset(self, student_sr_path, save_directory):
        answer_obj = RAG_answer()
        answer_obj.answer_dataset(student_sr_path, save_directory)





fire.Fire(RAG)  


