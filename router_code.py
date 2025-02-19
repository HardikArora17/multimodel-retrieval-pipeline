# !pip install langchain_openai langchain_huggingface

import os
import json
from typing import Literal
from typing_extensions import TypedDict
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_core.output_parsers import BaseOutputParser

from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

class RouteQuery(TypedDict):
    """Route query to destination."""
    destination: Literal[ "*Table-Type:*", "*Document-Type:*","*Image-Type:*"]

class CleanJsonOutputParser(BaseOutputParser):
    """Parses and sanitizes the LLM's JSON response."""

    def parse(self, text: str) -> RouteQuery:
        try:
            # Extract the first valid JSON block
            json_start = text.find("{")
            json_end = text.find("}") + 1

            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON object found in the output.")

            clean_text = text[json_start:json_end].strip()
            return json.loads(clean_text)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}. Output was: {text}")

def get_local_llama_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        eos_token_id=tokenizer.eos_token_id,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        do_sample=True,
        return_full_text=False,
        num_beams=3
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def get_route_chain():
  llm = get_local_llama_model('meta-llama/Meta-Llama-3-8B-Instruct')


  route_system = """
  You are an AI assistant responsible for routing user queries on high-performance computing systems. You have access to Frontier supercomputing system manuals for general document type queries. You also have access to to tables and images present wihin those documents.

  ### Task:
  Determine whether the user's query can be answered using the given documents textual part or if it requires tables present in the docs for answering or it requires an images as an answer.

  *Introduction:*
  When dealing with natural language questions, it's essential to discern whether the answer involves querying a structured database or retrieving information from unstructured or semi-structured documents. This classification helps in providing the most accurate and relevant response.

  *Instructions:*
  Analyze the following question using the criteria below:

  ### *Criteria for Table-Type Classification:*

  1. *Data Structure Implication:*
    - Does the question imply accessing data stored in tables or columns?
    - Are there references to specific data points like "average", "count", "max", suggesting aggregation or specific queries?

  2. *Relational Data:*
    - Is the question about relationships between different entities, suggesting joins or complex queries?
    - Mentions of terms like "records", "entries", "datasets" that typically relate to database records.

  3. *Specificity and Precision:*
    - Does it require exact matches or specific conditions in data retrieval (e.g., "all customers from New York with orders over $100")?

  4. *Temporal or Numerical Queries:*
    - Questions about trends over time, statistical analysis, or specific numerical conditions often point towards SQL queries.

  5. *Transactional Data:*
    - Inquiries about transactions, sales, inventory, or any business process that would typically be logged in a database.

  6. *Schema Knowledge:*
    - If the question assumes knowledge of a database schema or structure, it's likely SQL-type.

  ### *Criteria for Document-Type Classification:*

  1. *Unstructured or Semi-Structured Data:*
    - Does the question seek information that could be scattered across various documents or would require text search capabilities?

  2. *Content Type:*
    - References to manuals, guides, books, articles, or any narrative text where the answer might be embedded in paragraphs or sections.

  3. *Narrative or Descriptive Information:*
    - Questions looking for explanations, procedures, or conceptual understanding rather than data points.

  4. *Search and Retrieval:*
    - If the answer involves searching through text, looking for keywords, or understanding context from documents.

  5. *Knowledge Base or FAQ Style:*
    - Questions that could be answered from a knowledge base, help documentation, or FAQ sections.

  6. *Flexibility in Response:*
    - When the question allows for a broad interpretation or where the information might not be standardized or tabular.

  ### *Criteria for Image-Type Classification:*  

  1. *Visual Representation Required:*  
    - When an image is explicitly requested or would enhance understanding better than text.  

  2. *Concepts Best Understood Visually:*  
    - When the topic involves spatial relationships, structures, or patterns that are easier to grasp through images.  

  3. *Illustration, Recognition, or Data Visualization:*  
    - When the response involves drawings, object identification, or graphical representation of data.

  ### *Question Analysis:*

  - *Read the Question:* Carefully interpret the question, noting any implicit or explicit requests for data or information.
  - *Identify Keywords:* Look for terms that signal either the need for database queries or document search or image search (e.g., "database" vs. "manual").
  - *Context and Intent:* Determine the user's intent. Is it to get precise data or to understand a concept or process?

  ### *Classification:*

  - *Table-Type:* If the question predominantly matches criteria for structured table queries, classify as Table-Type. This includes scenarios where the answer involves data manipulation, aggregation, or relationship exploration within a table in documents.

  - *Document-Type:* If the question aligns more with retrieving information from texts form of documentation, classify it as document-type. This is often the case for questions seeking explanations, procedures, or where the data is less structured or more narrative.

  - *Image-Type:* If elements of both are present or if the question doesn't clearly fit either category, consider if it might require both approaches or if clarification from the user is necessary.

  *Final Step:*

  Based on your classification, provide a response tailored to the identified type, using relevant examples, explanations, or suggesting the appropriate retrieval method.

  
  ### Output format:
  - Respond **ONLY** in valid JSON format.
  - Do not include extra text, explanations, or prefixes.
  - Use the following schema:

  {{
    "destination": "*Table-Type:*" | "*Document-Type:*" | "*Image-Type:*"
  }}
  """

  template_messages = [
      SystemMessage(content=route_system),
      HumanMessagePromptTemplate.from_template("{text}"),
  ]

  route_prompt = ChatPromptTemplate.from_messages(template_messages)

  route_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", route_system),
          ("human", "{query}"),
      ]
  )

  parser = CleanJsonOutputParser()
  #parser = JsonOutputParser(pydantic_object=RouteQuery)

  route_chain = (
      route_prompt
      | llm
  )
  return route_chain, parser

