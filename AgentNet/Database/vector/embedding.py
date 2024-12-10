# here provides multiple embedding options 
from openai import OpenAI
import pandas as pd
#import faiss
import numpy as np

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


# def vector_search(query: str,
#                   collection,
#                   k=5):
#     data = pd.DataFrame(list(collection.find()), columns=['embedding', "_id"])
#     embeddings = np.array(data['embedding'].tolist())
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     query_embedding = get_embedding(query)
#     D, I = index.search(np.array([query_embedding]), k)
#     list_ids = data.iloc[I[0]]['_id'].tolist()
#     return list_ids
def vector_search(query: str,
                  collection,
                  k=5):
   raise NotImplementedError("This function is not implemented yet. Please use the `get_embedding` function to get the embedding of the query and then use the `faiss` library to search for the most similar embeddings in the collection.")
