import chromadb
from router_code.py import get_route_chain
from database import initialize_dbs, create_database

client = chromadb.PersistentClient(path="./chroma_db")
text_db, image_db, table_db = initialize_dbs(client)
text_db_1, image_db_1, table_db_1 = create_database(text_db, image_db, table_db, document_folder='/content/drive/MyDrive/RAG/output_files/documents')

def get_collection_count(collection):
    """Returns the number of records in a ChromaDB collection."""
    try:
        return collection.count()
    except Exception as e:
        print(f"Error retrieving count: {e}")
        return None

# Example usage
text_count = get_collection_count(text_db_1.collection)
print(f"Text Collection Count: {text_count}")

image_count = get_collection_count(image_db_1.collection)
print(f"Image Collection Count: {image_count}")

table_count = get_collection_count(table_db_1.collection)
print(f"Table Collection Count: {table_count}")

route_chain, parser = get_route_chain()
response = route_chain.invoke({"query": "what is number of nodes in setup?"})
json_response = parser.parse(response)['destination']

#======================================
#if conditions for json response
#=======================================

#FOR TABLE QUERY EXAMPLE
# query = 'what is batch job'
# table_str = [row.split('|') for row in table_db_1.answer_query(query, "")[0].split('\n')]
# df = pd.DataFrame(table_str[1:], columns=table_str[0])
# print(df)

#FOR TEXT QUERY EXAMPLE
# query = 'what is batch job'
# result = table_db_1.answer_query(query, "")

#FOR IMAGE QUERY EXAMPLE
# query = 'show me image of frontier gpu archiecture'
# result = image_db_1.answer_query(query, "")
