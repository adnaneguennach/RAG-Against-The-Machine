import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from search import RAG_search


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
    

    def answer(self, query):
        # print()
        search_obj = RAG_search(5)
        search_data = search_obj.search(query)
        return search_data


obj = QwenChat()
print(obj.answer("What activation formats does the fused batched MoE layer return in vLLM?"))