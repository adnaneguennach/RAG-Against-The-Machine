import fire
from search import RAG_search
from indexer import RAG_Indexer



class RAG:
    def __init__(self):
        pass
    
    #search 
    def index(self, max_chunk_size):
        index_obj = RAG_Indexer(max_chunk_size = max_chunk_size)
        raw_content, min_list = index_obj.index_corpus()
        index_obj.build_save_idx(raw_content, min_list)

    def search(self, query, k):
        search_obj = RAG_search(k = k)
        x = search_obj.search(query = query)
        # print(x)

    def search_dataset(self,dataset_path, k, save_directory):
        search_obj = RAG_search(k = k)
        search_obj.search_dataset(dataset_path, save_directory)

    # def answer(self, query):





fire.Fire(RAG)  


