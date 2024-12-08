from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from typing import List, Dict,Optional, Annotated
from AgentNet.Database.vector import get_embedding, vector_search
from uuid import uuid4
from AgentNet.config import nosql_service
from bson.objectid import ObjectId


PyObjectId = Annotated[str, BeforeValidator(str)]

class Node(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    label: str = Field(..., example="Node 1")
    description: str = Field(..., example="Description of Node 1")
    embedding: List[float] = Field(..., example=[0.1, 0.2, 0.3])
    type: Optional[str] = Field(None, example="type1")
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="allow",
        json_schema_extra={
            "example": {
                "label": "Node 1",
                "description": "Description of Node 1",
                "embedding": [0.1, 0.2, 0.3]
                 }})


class Edge(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    from_: str = Field("label", example="Node1", alias="from")
    to: str = Field("label", example="Node2")
    weight: Optional[float] = Field(description="measure of relavence, between 0-1",default=1)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="allow",
        json_schema_extra={
            "example": {
                "from": "Node1",
                "to": "Node2",
                "weight": 0.5
        }})

class NodeManager():
    def __init__(self, 
                 project):
        self.db = nosql_service[project]
    
    def create(self,                 
               label: str, 
               description: str,
               deduplicate_rule:list = ["label"],
               **kwargs) -> Node:
        """
        Create a new node.

        Args:
            label (_type_): _description_
            description (_type_): _description

        Returns:
            _type_: _description_
        """         
        all_inputs = {**kwargs, "label": label, "description": description}
        # check for duplicate first 
        dedupe_query = {rule: all_inputs[rule] for rule in deduplicate_rule}
        node = self.db["nodes"].find_one(dedupe_query)
        if node:
            print(f"Node with label {label} already exists")
            return Node(**node)
        else:
            embedding = get_embedding(f"label:{label}, description:{description}")
            node = Node(label=label, description=description, embedding=embedding, **kwargs)
            new_node = self.db["nodes"].insert_one(node.model_dump(by_alias=True, exclude=["id"]))
            node.id = new_node.inserted_id
            return node

    def delete(self,
               label: str):
        self.db["nodes"].delete_one({"label": label})
        return True

    def get(self,
            id:str):
        id = ObjectId(id)
        args = self.db["nodes"].find_one({"_id": id})
        if args:
            node = Node(**args)
            return node
        else:
            raise ValueError("Node not found")

    def get_all(self):
        args = self.db["nodes"].find()
        return [Node(**node) for node in args]
    
    def search_by_label(self,
               label):
        """Find all nodes with the given label.

        Args:
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        nodes = self.db["nodes"].find({"label": label})
        return [Node(**node) for node in nodes]
    
    def similarity_search(self,
                      query,
                      k=5):
        """Find all nodes with the given vector.

        Args:
            query (_type_): text query

        Returns:
            _type_: _description_
        """
        ids_list = vector_search(query, 
                                 collection=self.db["nodes"], 
                                 k = k)
        # remove duplicates
        ids_list = list(set(ids_list))
    
        return [self.get(id=id) for id in ids_list]
    
    def _get_nearest_neighbors(self,label:str):
        """find the nearest neighbors of a node that is connected to it by an edge."""
        edges = self.db["edges"].find({"$or": [{"from": label}, {"to": label}]})
        neighbors = []
        for edge in edges:
            if edge["from"] == label:
                neighbors.append(self.get(edge["to"]))
            else:
                neighbors.append(self.get(edge["from"]))
        return neighbors


    def get_neighbors(self, label: str, connection_degree: int = 1) -> dict:
        """Retrieve all nodes connected to a given node up to a certain degree.

        Args:
            label (str): The label of the starting node.
            connection_degree (int): Level of connection to retrieve.

        Returns:
            dict: A dictionary containing the starting node and its neighbors.
        """
        # use self.get_nearest_neighbors several times
        neighbors = {}
        current_level = 1
        labels = [label]
        while current_level <= connection_degree:
            for label in labels:
                neighbors[current_level] = self._get_nearest_neighbors(label)
            labels = [node.id for node in neighbors[current_level]]
            current_level += 1
        return neighbors
    
    def _get_nearest_neighbors_by_weight(self,labels:str):
        """find the nearest neighbors of a node that is connected to it by an edge but with probability."""
        import random
        edges = self.db["edges"].find({"$or": [{"from": labels}, {"to": labels}]})
        # do a random choice based on the weight for all the edges
        edges_weight = [edge.weight for edge in edges]
        # output the new edge based on the weight choosed by porbability
        def random_booleans(weights):
            # Normalize the weights to probabilities (if they are not already)
            probabilities = [weight / sum(weights) for weight in weights]
            
            # Generate a list of True/False based on the probabilities
            result = [random.random() < prob for prob in probabilities]
            
            return result
        def get_true_indices(boolean_list):
            return [index for index, value in enumerate(boolean_list) if value] 
        boolean_list = random_booleans(edges_weight)
        true_indices = get_true_indices(boolean_list)
        new_edges = [edges[i] for i in true_indices]
        neighbors = []
        for edge in new_edges:
            if edge["from"] == labels:
                neighbors.append(self.get(edge["to"]))
            else:
                neighbors.append(self.get(edge["from"]))
        return neighbors


    def get_neighbors_by_weight(self, label: str, connection_degree: int = 1) -> dict:
        """Retrieve all nodes connected to a given node up to a certain degree by weight

        Args:
            id (str): The ID of the starting node.
            connection_degree (int): Level of connection to retrieve.

        Returns:
            dict: A dictionary containing the starting node and its neighbors.
        """
        neighbors = {}
        current_level = 1
        labels = [label]
        while current_level <= connection_degree:
            for label in labels:
                neighbors[current_level] = self._get_nearest_neighbors_by_weight(label)
            labels = [node.label for node in neighbors[current_level]]
            current_level += 1
        return neighbors




class EdgeManager():
    def __init__(self, project):
        self.db = nosql_service[project]

    def create(self, from_label: str, to_label: str, weight: float,deduplicate_rule:list=["from","to"], **kwargs) -> Edge:
        all_inputs  = {**kwargs, "from": from_label, "to": to_label, "weight": weight}
        dedupe_query = {rule: all_inputs[rule] for rule in deduplicate_rule}
        # check if the edge already exists
        edge = self.db["edges"].find_one(all_inputs)
        if edge:
            print(f"Edge from {from_label} to {to_label} already exists")
            return Edge(**edge)
        else:
            args = Edge(from_=from_label, to=to_label, weight=weight, **kwargs)
            self.db["edges"].insert_one(args.model_dump(by_alias=True, exclude={"id"}))
            print(f"Edge from {from_label} to {to_label} created")
            return args

    def delete(self, from_label: str,to_label:str) -> bool:
        result = self.db["edges"].delete_one({"from": from_label, "to": to_label})
        return result.deleted_count > 0

    def get(self, id: str) -> Edge:
        object_id = ObjectId(id)
        edge = self.db["edges"].find_one({"_id": object_id})
        if edge:
            return Edge(**edge)
        else:
            raise ValueError("Edge not found")

    def get_all(self) -> List[Edge]:
        edges = self.db["edges"].find()
        return [Edge(**edge) for edge in edges]
