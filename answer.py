import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from search import RAG_search
from pathlib import Path
import json
from tqdm import tqdm
from models import StudentSearchResultsAndAnswer, MinimalAnswer

class QwenChat:
    def __init__(
        self, model_name="Qwen/Qwen3-0.6B", enable_thinking=False, max_new_tokens=2048
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate(self, context, query):
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the context. "
                    "Be concise. If the context is insufficient, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.max_new_tokens
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        return content
    

    def answer_query(self, query):
        search_obj = RAG_search(5)
        search_data = search_obj.search(query)
        context = ""
        for i, chunk in enumerate(search_data):
            path = Path(chunk["file_path"])
            content = path.read_text()
            dt = content[chunk["first_character_index"]:chunk["last_character_index"]]
            context += f"CONTEXT Number {i} : {dt}\n\n"

        answer = self.generate(context, query)
        return answer
    
    def answer_dataset(self,student_sr_path, save_directory):
        read_path = Path(student_sr_path)
        queries = []
        data = {}

        
        with open(student_sr_path, "r") as f:
            data = json.load(f)

        min_answer = []
        for sr in tqdm(data["search_results"]):
            question = sr["question"]
            min_lst = sr["retrieved_sources"]
            context = ""
            for i, chunk in enumerate(min_lst):
                chunk_path = Path(chunk["file_path"])
                context += f"CONTEXT NUMBER {i} : {chunk_path.read_text()[chunk['first_character_index']:chunk['last_character_index']]}\n\n"
            answer_data = self.generate(context, question)
            min_answer.append(MinimalAnswer(question_id = sr["question_id"], question = sr["question"] , retrieved_sources = sr["retrieved_sources"],answer = answer_data))

        save_data = StudentSearchResultsAndAnswer(
            search_results = min_answer,
            k = 5
        )



        with open("data/output/answer.json", "w") as f:
            json.dump(save_data.model_dump(), f, indent = 4)
        

    def evaluate(self, student_path, truth_path):
        import json
        
        with open(student_path) as f: student_data = json.load(f)
        with open(truth_path) as f: truth_data = json.load(f)
        
        truth_dict = {q["question_id"]: q.get("sources", []) for q in truth_data["rag_questions"]}
        
        recalls = {1: 0, 3: 0, 5: 0, 10: 0}
        total_qs = 0
        
        for ans in student_data["search_results"]:
            truth_sources = truth_dict.get(ans["question_id"])
            if not truth_sources: 
                continue
                
            total_qs += 1
            
            for t in truth_sources:
                for i, s in enumerate(ans["retrieved_sources"]):
                    if t["file_path"] != s["file_path"]: 
                        continue
                    
                    inter = max(0, min(t["last_character_index"], s["last_character_index"]) - max(t["first_character_index"], s["first_character_index"]))
                    union = max(t["last_character_index"], s["last_character_index"]) - min(t["first_character_index"], s["first_character_index"])
                    
                    if union > 0 and (inter / union) >= 0.05:
                        for k in recalls:
                            if i < k: 
                                recalls[k] += (1 / len(truth_sources))
                        break 
                        
        if total_qs == 0:
            return print("No valid questions found.")
            
        for k, score in recalls.items():
            print(f"Recall@{k}: {score / total_qs:.1%}")

obj = QwenChat()
# print(obj.answer_query("What activation formats does the fused batched MoE layer return in vLLM?"))
# print(obj.answer_dataset("datasets_public/public/UnansweredQuestions/dataset_code_public.json", None))
# print(obj.answer_dataset("data/output/test.json", None))
obj.evaluate(
    student_path="data/output/answer.json", 
    truth_path="datasets_public/public/AnsweredQuestions/dataset_code_public.json"
)