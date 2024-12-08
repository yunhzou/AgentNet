from pymongo import MongoClient
# dot env
from dotenv import load_dotenv
import os 
def connect_to_mongodb(uri):
    client = MongoClient(uri)
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)
        raise Exception("Unable to connect to the MongoDB deployment. Check your URI")        