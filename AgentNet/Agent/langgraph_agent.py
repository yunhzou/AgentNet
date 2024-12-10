from typing import Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph.graph import CompiledGraph
from .langgraph_supporter import LangGraphSupporter
from langgraph.graph.message import add_messages
from operator import add
from pydantic import BaseModel, Field
from typing import Annotated, Union, List
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage
from .agent_utils import get_date

class LangGraphAgent(LangGraphSupporter):
    class AgentState(BaseModel):
        messages: Annotated[list, add_messages]
        tool_used: Annotated[list, add]
    
    graph_state = AgentState

    def __init__(self,
                 model:str,
                 session_id: Optional[str] = "AgentSession" + get_date(),
                 action_logging_session_id: Optional[str] = "ActionSession" + get_date(),
                 interrupt_before: List[str]=[],
                 **kwargs
                 ):
        """
        Text completion in json mode

        Args:
            client (_type_): OpenAI client
            model (_type_): choose from openai exisiting models
            usersetup (_type_): user instructions that the assitant will always follow
        """
        self.llm = ChatOpenAI(temperature=0, model=model, **kwargs)
        self.interrupt_before = interrupt_before
        super().__init__(session_id=session_id)


    def _initialize_system_message(self):
        system_message = """Try to use tools you have to try to solve the CURRENT TASK. 
        If you do not have tools, you reject the task. 
        If your tool cannot solve it, you end the conversation. 
        You only work on the current task being given. 
        You should not work more than that even you know what to do. 
        Especially the task/objective in the CONTEXT should never be considered.
        
        Example:
        Context: TASK 1, 2, 3,4,5
        Current Task: TASK 9

        You should only work on TASK 9!
        """ 
        return self.rewrite_system_message(system_message)

    def _create_agent(self)->CompiledGraph:
        workflow = StateGraph(self.graph_state)
        tool_node = ToolNode(self.tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools,parallel_tool_calls=False)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        self.agent = workflow.compile(checkpointer=self.memory_manager,interrupt_before=self.interrupt_before)

    def should_continue(self,state: AgentState):
        messages = state.messages
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    def call_model(self,state: AgentState):
        messages = state.messages
        response = self.llm_with_tools.invoke(messages)
        if response.tool_calls:
            if len(response.tool_calls) > 1:
                Warning("Multiple tools are called, this is not allowed in this design of agents")
                return {"messages": [response]}
            else:
                tool_call=response.tool_calls[0]
                tool_name:str = tool_call['name']
                args:dict = tool_call['args']
                tool_information = {"tool_name": tool_name, "args": args}
                return {"messages": [response],"tool_used": [tool_information]}

        return {"messages": [response]}

        