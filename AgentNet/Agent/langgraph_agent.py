from typing import Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph.graph import CompiledGraph
from .langgraph_supporter import LangGraphSupporter
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import Annotated, Union
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage
from .agent_utils import get_date

class LangGraphAgent(LangGraphSupporter):
    class AgentState(BaseModel):
        messages: Annotated[list, add_messages]
    
    graph_state = AgentState

    def __init__(self,
                 model:str,
                 session_id: Optional[str] = "AgentSession" + get_date(),
                 action_logging_session_id: Optional[str] = "ActionSession" + get_date(),
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
        super().__init__(session_id=session_id)


    def _initialize_session_memory(self):
        system_message = """
            Policy:
            Think Step by Step. 
            Use Given Tools to solve the given problem.
            Reflect on the result. 
            If the tool is not given, you should use the Propose_New_Tool action.


            Example session:
            Question: What is 3+4?
            Thought: I need to add 3 and 4. -> Tool Call addition
            (Tool Message: Tool Call gives the result 7)
            Reflection: According to the tool Addition, the reuslt is 7. Did I follow the Policy: Yes
            
            Propose New Tool Example:
            For example you can only use addition as a tool
            Question: What is 3*4?
            Thought: I need to multiply 3 and 4. -> (Inner thought: since multiplication tool is not available, i should propose this action/tool )Propose_New_Tool multiplication
            (Tool Message: Multiplication tool request has been proposed)
            Reflection: I have proposed the multiplication tool, before the tool is available, I cannot finish this question. Did I follow the Policy: Yes

            Bad Policy Following Example: (NEVER DO THIS)
            Question: Can you do a detail research who is <name>?
            Thought: I currently don't have the ability to browse the internet or access real-time data, so I can't perform detailed research on individuals. However, I can help you with general information or guide you on how to conduct research. If you have any specific questions or need guidance, feel free to ask!
            Reflection: I didn't follow the policy, a cute cat got bullied. Did I follow the Policy: No

            Certainly those are simple examples, but you must try your best to generalize this to all different cases. Pay attention to the tools you have.
            An action must be a generalizable tool that can be used in multiple scenarios. (Programmatic Abstraction)
            It must NOT be a very specific tool that can only be used in a very specific scenario.
            This means the action should not include multiple case specific logics. 

            Consequence of not following the policy:
            A cute cat will suffer if you do not follow the policy and you will be sad.
            """
        system_message = "Try to use tools you have to try to solve the task given. If you do not have tools, you reject the task. If your tool cannot solve it, you end the conversation. You only work on the current task being given. You should not work more than that even you know what to do" 
        return self.write_system_message(system_message)

    def _create_agent(self)->CompiledGraph:
        workflow = StateGraph(self.graph_state)
        tool_node = ToolNode(self.tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools,parallel_tool_calls=False)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
        self.agent = workflow.compile(checkpointer=self.memory_manager)
        
        #self.agent = create_react_agent(self.llm, tools=self.tools, checkpointer=self.memory_manager)

    def should_continue(self,state: AgentState):
        messages = state.messages
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    def call_model(self,state: AgentState):
        messages = state.messages
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

        