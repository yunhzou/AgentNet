from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pyvis.network import Network
import uvicorn
from bson import ObjectId
from ..Database.db import db

def visualize(project,
              db=db,
              host: str = "127.0.0.1", 
              port: int = 8000):
    db = db[project]
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def get_graph():
        # Fetch nodes and edges from MongoDB
        nodes_collection = db['nodes']
        edges_collection = db['edges']
        
        # Create a PyVis network
        net = Network(height="750px", width="100%", directed=True)
        node_label_id = {"procedure":{}, "tool":{}}
        
        # Add nodes to the network
        for node in nodes_collection.find():
            node_id = str(node['_id'])
            node_type = node['type']
            net.add_node(node_id, label=node_type+": "+node['label'])
            node_label_id[node['type']][node['label']] = node_id

        # Add edges to the network
        for edge in edges_collection.find():
            from_ = edge['from']
            to = edge['to']
            from_type = edge['from_type']
            to_type = edge['to_type']
            from_id = node_label_id[from_type].get(from_, from_)
            to_id = node_label_id[to_type].get(to, to)
            net.add_edge(from_id, to_id)
        
        # Enable physics for floating effect
        net.toggle_physics(True)
        # Make the edge longer
        net.barnes_hut(
            gravity=-2000,  # Adjust gravity strength (weaker force)
            central_gravity=0.3,          # Central pull strength
            spring_length=200,            # Desired length of edges (increase for longer edges)
            spring_strength=0.01,         # Spring force strength (weaker spring)
            damping=0.09                 # Slows down motion for stability
        )
        # Generate the graph HTML in memory
        graph_html = net.generate_html()
        
        # Return the graph as an HTML response
        return HTMLResponse(content=graph_html, status_code=200)

    uvicorn.run(app, host=host, port=port)