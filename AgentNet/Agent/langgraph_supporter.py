# abc class
from abc import ABC, abstractmethod
from typing import Optional, Union,List
from .langgraph_utils import MongoDBSaver
from pydantic import BaseModel, Field
from typing import Optional, Annotated, TypedDict, Union,List
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from AgentNet.config import nosql_service
from .agent_utils import image_to_base64, get_date

class LangGraphSupporter(ABC):
    class GeneralState(TypedDict):
        messages: Annotated[list, add_messages]
    history_key = "messages"
    graph_state = GeneralState
    system_message = None
    recursion_limit = 1000
    def __init__(self,session_id: Optional[str] = "AgentSession" + get_date(),**kwargs):
        self.memory_manager = MongoDBSaver(client = nosql_service, db_name = "chat_history")
        self.config = {"configurable": {"thread_id": session_id}, "recursion_limit": self.recursion_limit} 
        self.tools = []
        self._create_agent()
        self._initialize_session_memory()
        self.message_blocks = []

    @abstractmethod
    def _create_agent(self)->CompiledGraph:
        """Call this function to create the graph as agent,
            self.agent: CompiledGraph
        """
        workflow = StateGraph(self.graph_state)
        self.agent = workflow.compile(checkpointer=self.memory_manager)
        raise NotImplementedError
        
    def _initialize_session_memory(self):
        """
        Insert agent system message
        """
        system_message = "You are a helpful assistant!"
        self.write_system_message(system_message)


    def write_system_message(self,system_message:str):
            """
            Overwrite the systen message in the agent memory

            Args:
                system_message (str): The system message
            """
            # check if system message contains placeholder such as {}
            if "{}" in system_message:
                Exception("System message injection currently do not support placeholder, build your custom agent instead for this purpose")
            self.system_message = system_message
            past_state = self.agent.get_state(self.config).values
            if past_state == {}:
                self.agent.update_state(self.config, {"messages": [SystemMessage(content=system_message)]})
            else:
                # find the first system message in past_state['messages']
                for i, msg in enumerate(past_state['messages']):
                    if type(msg) == SystemMessage:
                        past_system_message = past_state['messages'][i]
                        break
                    else:
                        raise Exception("No system message found in the past state")
                past_system_message.content = system_message
                self.agent.update_state(self.config, {"messages": past_state['messages']})

    def add_switchable_text_block_system_message(self,
                                                 block_name:str,
                                                 block_description:str, 
                                                 block_content:str):
        # append a grounding message section to the system message. When this function is called, always remove the previous grounding message
        block_prefix = block_name + ":\n"
        block_message = f"\n{block_prefix}\n{block_description}\n{block_content}"
        # check if system message contains grounding message
        # TODO: this assumes grounding always comes after the system message
        if block_prefix in self.system_message:
            self.system_message = self.system_message.split(block_prefix)[0]
        self.system_message += block_message
        self.write_system_message(self.system_message)

    def add_switchable_text_block_message(self,
                                            block_name:str,
                                            block_description:str, 
                                            block_content:str):
            """ Add a switchable text block to the agent memory"""
            block_prefix = block_name + ":\n"
            block_message = f"\n{block_prefix}\n{block_description}\n{block_content}"
            self.message_blocks.append(block_message)


    def append_system_message(self,system_message:str):
        # check if system message contains placeholder such as {}
        if "{}" in system_message:
            Exception("System message injection currently do not support placeholder, build your custom agent instead for this purpose")
        self.system_message = self.system_message+system_message
        self.write_system_message(self.system_message)

    def add_tool(self,tool: Union[BaseTool, list]):
        if isinstance(tool, list):
            for t in tool:
                if t in self.tools:
                    continue
                self.tools.append(t)
        
        else:
            if tool in self.tools:
                print(f"tool already exists in the agent")
            else:   
                self.tools.append(tool)
        self._create_agent()

    def detach_tool(self):
        self.tools = []
        self._create_agent()

    def invoke(self,user_input,image_urls:Optional[List[str]]=None)->List[BaseMessage]:
        """image_urls: list of image urls, can be local dir"""
        if len(self.message_blocks)>0:
            additional_message = "\n".join(self.message_blocks) + "\n"
            user_input = user_input + additional_message
        if image_urls is None:
            image_urls = []
        image_data = [image_to_base64(image_url) for image_url in image_urls]
        user_input = HumanMessage(
            content=[{"type": "text", "text": user_input}] + [{"type": "image_url", "image_url": {"url": img}} for img in image_data]
        )
        response = self.agent.invoke({"messages":[user_input]},self.config)
        return response["messages"]
    
    def stream(self,user_input,image_urls:Optional[List[str]]=None)->List[BaseMessage]:
        """image_urls: list of image urls, can be local dir"""
        if len(self.message_blocks)>0:
            additional_message = "\n".join(self.message_blocks) + "\n"
            user_input = user_input + additional_message
        if image_urls is None:
            image_urls = []
        image_data = [image_to_base64(image_url) for image_url in image_urls]
        user_input = HumanMessage(
            content=[{"type": "text", "text": user_input}] + [{"type": "image_url", "image_url": {"url": img}} for img in image_data]
        )

        messages = []
        for event in self.agent.stream({"messages": [user_input]},config=self.config):
            for value in event.values():
                messages.append(value["messages"][-1])
                print(value["messages"][-1].content)
        return messages
    
    def invoke_return_state(self,user_input,image_urls:Optional[List[str]]=None)->dict:
        """invoke method but return the state"""
        if len(self.message_blocks)>0:
            additional_message = "\n".join(self.message_blocks) + "\n"
            user_input = user_input + additional_message
        if image_urls is None:
            image_urls = []
        image_data = [image_to_base64(image_url) for image_url in image_urls]
        user_input = HumanMessage(
            content=[{"type": "text", "text": user_input}] + [{"type": "image_url", "image_url": {"url": img}} for img in image_data]
        )
        response = self.agent.invoke({"messages":[user_input]},self.config)
        return response


    def clear_memory(self,print_log=True):
        thread_id = self.config["configurable"]["thread_id"]
        self.memory_manager.delete(thread_id,print_log=print_log)
        self._initialize_session_memory()
