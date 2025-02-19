import transformers
import torch
from typing import List

from .base_generator import BaseGenerator

class LlamaGenerator(BaseGenerator):
    def __init__(self, model_name, token) -> None:
        self.pipeline = transformers.pipeline(
                                                "text-generation",
                                                model=model_name,
                                                model_kwargs={"torch_dtype": torch.float16},
                                                device_map="auto",
                                                token=token
                                            )
        self.terminators = [
                        self.pipeline.tokenizer.eos_token_id,
                        self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

    def build_prompt(self, user_query: str, contexts: List[str]) -> str:
        messages =  [
                        {"role": "system",
                          "content": "Use the contexts provided below and answer the question following the contexts. The answer should be generated using the contexts only. If the contexts seems insufficient to answer the question respond with a message stating that question cannot be asnwered due to lack of information."
                          },
                        
                    ]

        query = "Contexts:\n"
        query += "\n".join([f'{i+1}. {context}' for i,context in enumerate(contexts)])
        query += f'\nQuestion: {user_query}'

        messages.append( {"role": "user", "content": query} )

        prompt = self.pipeline.tokenizer.apply_chat_template(
                                                            messages, 
                                                            tokenize=False, 
                                                            add_generation_prompt=True
                                                                    )
        
        return prompt
    
    def parse_response(self, model_output: str) -> str:
        response = model_output.split("<|end_header_id|>")[-1]
        return response.strip()
    
    def generate(self, query: str, contexts: List[str], **pipeline_kwargs) -> str:
        prompt = self.build_prompt(user_query=query, contexts=contexts)
        model_outputs = self.pipeline(
                            prompt,
                            eos_token_id=self.terminators,
                            **pipeline_kwargs
                        )
        model_outputs = model_outputs[0]['generated_text']
        response = self.parse_response(model_output=model_outputs)

        return response
        
