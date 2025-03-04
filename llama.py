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

    def build_text_prompt(self, user_query: str, contexts: List[str]) -> str:
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
    
    def build_table_prompt(self, user_query: str, contexts: List[str]) -> str:
        messages =  [
                        {"role": "system",
                          "content": "Use the table provided below and answer the question based on the table. The answer should be generated using the table only. If it seems insufficient to answer the question respond with a message stating that question cannot be asnwered due to lack of information."
                          },
                        
                    ]

        query = "Table:\n"
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
    
    def generate(self, query: str, contexts: List[str], query_type = 'document', **pipeline_kwargs) -> str:
        
        if query_type == 'table':
          prompt = self.build_table_prompt(user_query=query, contexts=contexts)
        else:
          prompt = self.build_text_prompt(user_query=query, contexts=contexts)

        model_outputs = self.pipeline(
                            prompt,
                            eos_token_id=self.terminators,
                            **pipeline_kwargs
                        )
        model_outputs = model_outputs[0]['generated_text']
        response = self.parse_response(model_output=model_outputs)

        return response
        
