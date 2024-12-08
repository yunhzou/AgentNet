# node would be agent or tool 
# edge would be the relationship between agents or tools
from typing import Union,List, Callable
from .Agent import LangGraphSupporter, LangGraphAgentReactExperimental, LangGraphAgent
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import tool
from AgentNet import NodeManager,EdgeManager
import uuid
import types
from AgentNet.Agent import LangGraphAgentCritic

"""
Relavent files needed:
Database: stores the node and connections of the agent network
Toolmap: stores the mapping between tool name and tool function, import all the actions there

Each Node is an agent or a tool
The endpoint will always be a tool
"""

def add_docstring(docstring, label):
    def decorator(func):
        func.__doc__ = docstring

        def wrapper(*args, **kwargs):
            print(f"[{label}] ...")
            result = func(*args, **kwargs)
            print(f"[{label}] Finished!")
            return result
        
        # Return the original function while maintaining the wrapped behavior
        wrapper.__doc__ = func.__doc__  # Preserve the docstring for the wrapper
        wrapper.__name__ = func.__name__  # Preserve the original function name
        wrapper.__module__ = func.__module__  # Preserve the original module
        return wrapper
    return decorator

class AgentNetToolNode(BaseModel):
    label: str
    description:str  # not used for now 
    entity: BaseTool
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True)

class AgentNetProcedureNode(BaseModel):
    label: str
    description:str
    entity: LangGraphSupporter
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True)
    def stream(self,query:str):
        print("Agent: ",self.label)
        return self.entity.stream(query)


class AgentNetProcedureEdge(BaseModel):
    from_: AgentNetProcedureNode =  Field(alias="from")
    to: Union[AgentNetToolNode,AgentNetProcedureNode]
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True)

class AgentNetManager():
    """Manages the agent network including nodes and edges."""
    def __init__(self,project):
        self.procedure_nodes:List[AgentNetProcedureNode] = []
        self.tool_nodes:List[AgentNetToolNode] = []
        self.edges: List[AgentNetProcedureEdge] = []
        self.node_manager = NodeManager(project)
        self.edge_manager = EdgeManager(project)
        self.initialize()   
        self.grounding_context=None
        self.grounding=None

    def add_grounding(self, grounding_context:str,grounding:Callable):
        self.grounding_context = grounding_context
        self.grounding = grounding

    def get_tool_map(self):
        """
        Acquire the tool map
        Tool map is in the format of {tool_name:tool}
        """
        from tool_map import tool_map
        return tool_map

    def _get_all_nodes_metadata(self):
        return self.node_manager.get_all()
    
    def _get_all_edges_metadata(self):
        return self.edge_manager.get_all()
    
    def create_procedure_node(self,
                              label:str,
                              description:str,
                              model="gpt-4o"):
        """Create a procedure node from metadata

        Args:
            label (str): label of the node
            description (str): description of the procedure node, usually says what the agent can do or should do 

        Returns:
            AgentNetProcedureNode: Procedure node
        """
        # TODO: Maybe Plug in Dspy for optimization
        entity = LangGraphAgentReactExperimental(model=model,session_id=label)
        entity.append_system_message(description)
        procedure_node = AgentNetProcedureNode(label=label,description=description,entity=entity)
        if procedure_node.label in [node.label for node in self.procedure_nodes]:
            print(f"Procedure node {procedure_node.label} already exists")
            return procedure_node
        self.procedure_nodes.append(procedure_node)
        return procedure_node
    
    def create_procedure_chat_node(self,label:str,
                              description:str):
        """Create a procedure chat node from metadata

        Args:
            label (str): label of the node
            description (str): description of the procedure node, usually says what the agent can do or should do 

        Returns:
            AgentNetProcedureNode: Procedure node
        """
        # TODO: Maybe Plug in Dspy for optimization
        entity = LangGraphAgent(model="gpt-4o",session_id=label+str(uuid.uuid4()))
        entity.append_system_message(description)
        procedure_node = AgentNetProcedureNode(label=label,description=description,entity=entity)
        if procedure_node.label in [node.label for node in self.procedure_nodes]:
            print(f"Procedure node {procedure_node.label} already exists")
            return procedure_node
        self.procedure_nodes.append(procedure_node)
        return procedure_node

    def create_tool_node(self,
                         label:str,
                         description:str):
        """Create a tool node from metadata

        Args:
            label (str): label of the tool, shold always be the exact tool name (function name)
            description (str): description of what the tool does. For now this info is not used by the pipeline as it overlaps with the original tool schema documentation

        Raises:
            ValueError: _description_

        Returns:
            AgentNetToolNode: Tool node
        """
        entity = self.tool_map[label]
        if entity is None:
            raise ValueError(f"Tool {label} not found in tool map")
        tool_node = AgentNetToolNode(label=label,description=description,entity=entity)
        if tool_node in self.tool_nodes:
            print(f"Tool node {tool_node} already exists")
            return tool_node
        self.tool_nodes.append(tool_node)
        return tool_node

    def create_edge(self,
                    from_:Union[str,AgentNetProcedureNode],
                    to:Union[AgentNetToolNode,AgentNetProcedureNode,str]):
        """Create an edge between two nodes

        Args:
            from_ (AgentNetProcedureNode): Source node
            to (Union[AgentNetToolNode,AgentNetProcedureNode]): Destination node

        Returns:
            AgentNetProcedureEdge: Edge
        """
        if isinstance(from_,str):
            from_ = next(node for node in self.procedure_nodes if node.label==from_)
        if isinstance(to,str):
            to = next(node for node in self.procedure_nodes+self.tool_nodes if node.label==to)
        edge = AgentNetProcedureEdge(from_=from_,to=to)
        # check for duplicate
        if edge in self.edges: 
            print(f"Edge {edge} already exists")
            return edge
        self.edges.append(edge)
        return edge

    def initialize(self):
        """
        Initialize the agent network from the database
        """
        self.tool_map = self.get_tool_map()
        nodes_meta = self._get_all_nodes_metadata()
        edges_meta = self._get_all_edges_metadata()
        node_dict = {"procedure":{}, "tool":{}}

        for node_meta in nodes_meta:
            if node_meta.type == "tool":
                node = self.create_tool_node(label=node_meta.label, description=node_meta.description)
            elif node_meta.type == "procedure":
                node = self.create_procedure_node(label=node_meta.label, description=node_meta.description)
            node_dict[node_meta.type][node.label] = node

        for edge_meta in edges_meta:
            from_node = node_dict[edge_meta.from_type].get(edge_meta.from_)
            to_node = node_dict[edge_meta.to_type].get(edge_meta.to)
            if from_node and to_node:
                edge = AgentNetProcedureEdge(from_=from_node, to=to_node)
                if edge in self.edges:
                    print(f"Edge {edge} already exists")
                else:
                    self.edges.append(edge)
            else:
                print(f"Warning: Node not found for edge {edge_meta}")

    def connect_node(self,node:AgentNetProcedureNode)->AgentNetProcedureNode:
        """
        Connect a single node to its neighbors, i.e. add tools to the agent, add agent as tools to the agent
        """
        edges_for_node = [edge for edge in self.edges if edge.from_.label == node.label]
        nodes_to_connect = [edge.to for edge in edges_for_node]
        tools_to_add = [
            self.agent2tool(node_to_connect.entity,node_to_connect.description+"the query field must be a description string not any object",node_to_connect.label) if isinstance(node_to_connect, AgentNetProcedureNode) else node_to_connect.entity
            for node_to_connect in nodes_to_connect
        ]
        node.entity.add_tool(tools_to_add)
        tools_description = ["Action: "+ node_to_connect.label + " Action Description: " + node_to_connect.description for node_to_connect in nodes_to_connect]
        tools_description = "\n Here are the Available actions you have access to: \n"+"\n".join(tools_description) + "\n Try use them when proposing the actions. Say the action name in objective section and pass the action necessary input context. This is very important and if you do not follow this rule, you will get punished! \n You should Terminate if your tool cannot finish the task."

        node.entity.append_system_message(tools_description)
        return node

    def compile(self):
        """
        Compile the agent network, equip all the nodes with its connections
        """

        for node in self.procedure_nodes:
            node.entity.detach_tool()
            self.connect_node(node)
            print(f"Node {node.label} connected")
        print("Compilation complete")

    def store_node(self,node:Union[AgentNetProcedureNode,AgentNetToolNode]):
        """
        Store a single node in the database

        Args:
            node (Union[AgentNetProcedureNode,AgentNetToolNode]): Node to store
        """
        if type(node)==AgentNetToolNode:
            self.node_manager.create(label=node.label,description=node.description,type="tool",deduplicate_rule=["label","type"])
        elif type(node)==AgentNetProcedureNode:
            self.node_manager.create(label=node.label,description=node.description,type="procedure",deduplicate_rule=["label","type"])
    
    def store_edge(self,edge:AgentNetProcedureEdge):
        """
        Store a single edge in the database

        Args:
            edge (AgentNetProcedureEdge): Edge to store
        """
        from_type = "procedure" if isinstance(edge.from_,AgentNetProcedureNode) else "tool"
        to_type = "procedure" if isinstance(edge.to,AgentNetProcedureNode) else "tool"
        self.edge_manager.create(from_label=edge.from_.label,to_label=edge.to.label,from_type=from_type, to_type=to_type, weight=1)

    def store(self):
        """
        Store all the nodes and edges in the database
        """
        for node in self.procedure_nodes+self.tool_nodes:
            self.store_node(node)
        for edge in self.edges:
            self.store_edge(edge)

    def stream(self,input_node_label:str,query:str):
        """
        Stream the input query through the agent network

        Args:
            input_node_label (str): label of the input node
            query (str): input query

        Returns:
            str: output of the agent network
        """
        input_node = next(node for node in self.procedure_nodes if node.label==input_node_label)
        # update grounding
        # TODO: add critic to main agent
        if self.grounding:
            updated_grounding = self.grounding()
            #input_node.entity.add_switchable_text_block_system_message(block_name="Grounding",block_description=self.grounding_context,block_content=updated_grounding)
            input_node.entity.add_switchable_text_block_message(block_name="Grounding",block_description=self.grounding_context,block_content=updated_grounding)
        return input_node.stream(query)

    def agent2tool(self,agent:LangGraphSupporter, description: str, label: str) -> BaseTool:
        """Convert an agent to a tool with a dynamic name."""
        # Define the inner function with the desired logic
        @add_docstring(description, label)
        def dynamic_tool(query: str):
            if self.grounding:
                updated_grounding = self.grounding()
                agent.add_switchable_text_block_message(block_name="Grounding",block_description=self.grounding_context,block_content=updated_grounding)
            conversation = agent.stream(query)
            result_agent_response = conversation[-1].content
            result_observation = conversation[-2].content
            """Adding critic agent"""
            result = result_agent_response + "\n" + result_observation
            critic_agent =LangGraphAgentCritic(model="gpt-4o",session_id="criticagent_test")
            critic_input = "Task: " + query + "\n" + "Agent Response: " + result_agent_response + "\n" + "Agent Observation: " + result_observation
            critic_result = critic_agent.invoke_return_state(critic_input)
            critic_reasoning = critic_result["critic"]
            critic_result = critic_result["success"]
            print(f"\033[92mCritic Result: {critic_result}\033[0m")
            print(f"\033[92mCritic Reasoning: {critic_reasoning}\033[0m")
            if critic_result!=True:
                human_intervention=input("The agent failed to finish the task. To Continue, type Y, to raise Error, type N") 
                if human_intervention=="Y":
                    return result
                else:
                    raise ValueError(f"Agent failed to finish the task. Critic Reasoning: {critic_reasoning}")
            #agent.clear_memory()
            return result

        # Dynamically set the function name using the `types.FunctionType` constructor
        dynamic_tool_with_label = types.FunctionType(
            dynamic_tool.__code__,
            dynamic_tool.__globals__,
            name=label,  # Set the desired function name
            argdefs=dynamic_tool.__defaults__,
            closure=dynamic_tool.__closure__
        )

        # Transfer metadata like docstring
        class query(BaseModel):
            query: str = Field(description="The full CONTEXT of the task you try to perform. You are talking to a AGENT, not a function. Be detail and specific")
        dynamic_tool_with_label.__doc__ = dynamic_tool.__doc__
        dynamic_tool_with_label = tool(dynamic_tool_with_label,args_schema=query)  # Reapply the tool decorator

        return dynamic_tool_with_label