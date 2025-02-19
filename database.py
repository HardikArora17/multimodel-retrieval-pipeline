import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPProcessor, CLIPModel
import json
from PIL import Image
from langchain.text_splitter import MarkdownTextSplitter
from openai import OpenAI
import json
import pandas as pd
from tqdm.autonotebook import tqdm
import chromadb.utils.embedding_functions as embedding_functions
import os
from docutils.core import publish_doctree
from retrieval.mxbai_retriever import MxbaiRetriever
from reranker.mxbai_reranker import MxbaiReranker
from generator.llama import LlamaGenerator
from generator.phi import PhiGenerator
from vectordb.chromadb import ChromaDB

class TextDatabase:
    def __init__(self, client, collection):
        self.collection = collection
        self.text_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True)
        self.splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50)
        self.retriever = MxbaiRetriever('mixedbread-ai/mxbai-embed-large-v1',collection)
        self.ranker = MxbaiReranker('mixedbread-ai/mxbai-rerank-large-v1')
        self.generator = LlamaGenerator('meta-llama/Llama-3.2-1B')
    
    def index(self, text):
        chunks = self.splitter.split_text(text)
        vectors = [self.text_model.encode(chunk).tolist() for chunk in chunks]
        metadata = [{"chunk": chunk} for chunk in chunks]
        ids = [str(len(self.collection.get()["ids"]) + i + 1) for i in range(len(chunks))]
        self.collection.add(embeddings=vectors, metadatas=metadata, ids=ids)
      
    def answer_query(self, query, client):
      #result = self.retriever.retrieve(collection_name='text_v1', query=[query], topk=9)
      query_vector = self.text_model.encode(query).tolist()
      result = self.collection.query(query_embeddings=[query_vector], n_results=9)
      retrieved_chunks = list(set([chunk_dict['chunk'] for chunk_dict in result['metadatas'][0]]))
      ranked_result = self.ranker.rank(query, retrieved_chunks, topk=3)
      answer = self.generator.generate(query, ranked_result, max_new_tokens=1024, top_p=0.9, temperature=0.1)
      return ""

# Image Database Class
class ImageDatabase:
    def __init__(self, client, collection):
        self.collection = collection
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def add_data(self, images):
        metadata_list = []
        vector_list = []
        ids = []
        for i, image_path in enumerate(images, start = (len(self.collection.get()["ids"])+1)):
          image = Image.open(image_path)
          inputs = self.clip_processor(images=[image], return_tensors="pt")
          with torch.no_grad():
             vector_list.append(self.clip_model.get_image_features(**inputs).squeeze().tolist())
          metadata_list.append({"path": image_path})
          ids.append(str(i))

        self.collection.add(embeddings=vector_list, metadatas=metadata_list, ids=ids)

    def encode_clip_text(self,text):
      inputs = self.clip_processor(text=[text], return_tensors="pt")
      with torch.no_grad():
          vector = self.clip_model.get_text_features(**inputs).squeeze().tolist()
      return vector

    def answer_query(self,text_query):
        query_vector = self.encode_clip_text(text_query)
        search_result = self.collections.query(query_embeddings=[query_vector], n_results=1)
        return search_result if search_result else "No relevant image found"

# Table Database Class
class TableDatabase:
    def __init__(self, client, collection):
        self.collection = collection
        self.text_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True)
        self.retriever = MxbaiRetriever('mixedbread-ai/mxbai-embed-large-v1',collection)
        self.ranker = MxbaiReranker('mixedbread-ai/mxbai-rerank-large-v1')
    
    def encode_table_as_text(self, table):

      table_str = "\n".join([(" | ".join(map(str, row))).replace("\n","") for row in table])  # Convert to readable text
      return table_str

    def index(self, table):
        table_str = self.encode_table_as_text(table)
        vector = self.text_model.encode(table_str).tolist()
        metadata = {"table": table_str}
        self.collection.add(embeddings=[vector], metadatas=[metadata], ids=[str(len(self.collection.get()["ids"]) + 1)])

    def answer_query(self, query, client):
      query_vector = self.text_model.encode(query).tolist()
      result = self.collection.query(query_embeddings=[query_vector], n_results=1)
      retrieved_chunks = list(set([chunk_dict['table'] for chunk_dict in result['metadatas'][0]]))
      return retrieved_chunks


def initialize_dbs(client):
  # try:
  #   client.delete_collection("text_v1")
  # except:
  #   pass

  # try:
  #   client.delete_collection("images_v1")
  # except:
  #   pass

  # try:
  #   client.delete_collection("tables_v1")
  # except:
  #   pass


  text_ef = embedding_functions.HuggingFaceEmbeddingFunction(
      api_key = "",
      model_name="mixedbread-ai/mxbai-embed-large-v1"
  )

  # Define collections
  collections = {
      "text": client.get_or_create_collection("text_v1", embedding_function =text_ef),
      "images": client.get_or_create_collection("images_v1"),
      "tables": client.get_or_create_collection("tables_v1")
  }

  text_db = TextDatabase(client, collections["text"])
  image_db = ImageDatabase(client, collections["images"])
  table_db = TableDatabase(client, collections["tables"])
  return text_db, image_db, table_db

def create_database(text_db, image_db, table_db, document_folder = '/content/documents'):
  count=0
  for file_name in os.listdir(document_folder):
      if not file_name.endswith(".rst"):
        continue

      file_path = os.path.join(document_folder, file_name)

      with open(file_path, 'r', encoding='utf-8') as file:
          rst_content = file.read()

      doctree = publish_doctree(rst_content)

      text_content = []
      tables = []
      images = []

      def is_inside_table(node):
          current_node = node
          while current_node.parent is not None:
              if hasattr(current_node.parent, 'tagname') and current_node.parent.tagname == 'table':
                  return True
              current_node = current_node.parent
          return False

      for node in doctree.traverse():
          if node.tagname == 'paragraph':
              if not is_inside_table(node):
                  text_content.append(node.astext())
          elif node.tagname == 'table':
              rows = []
              for row in node.traverse(condition=lambda x: x.tagname == 'row'):
                  cells = [cell.astext() for cell in row.traverse(condition=lambda x: x.tagname == 'entry')]
                  rows.append(cells)
              tables.append(rows)
          elif node.tagname == 'image':
              if 'uri' in node.attributes:
                  images.append(node.attributes['uri'][1:])


      output_dict = {}
      for i, text in tqdm(enumerate(text_content)):
        text_db.index(text)
        if i>50:
          break
      
      if len(images)>0:
        image_db.add_data(images)
        print(images)

      for i, table in tqdm(enumerate(tables)):
        table_db.index(table)
      

      count+=1
      if count>1:
        break
    
  return text_db, image_db, table_db
  
