from pydantic import BaseModel, PrivateAttr
from pymongo import MongoClient

# MongoDB client setup
nosql_service = MongoClient("mongodb://localhost:27017/")["my_database"]

class MyModel(BaseModel):
    field1: int
    field2: str

    def model_post_init(self, __context) -> None:
        # Convert the Pydantic model to a dictionary and insert it into MongoDB
        document = self.model_dump()
        nosql_service['model'].insert_one(document)
        print(f"Inserted document: {document}")

# Example usage
my_instance = MyModel(field1=123, field2="example")