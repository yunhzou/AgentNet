from .langgraph_supporter import LangGraphSupporter 
from .langgraph_agent import LangGraphAgent
from pydantic import BaseModel, Field
from typing import Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI 
import uuid
from .agent_utils import get_date

class ReACT(BaseModel):
    thought: str = Field(description="One thought process.Each thought is only one sentence and should have a objective, you must follow this rule")
    action: str = Field(description="An action should be taken to achieve the thought, only one action is allowed. Describe the action in one sentence, best to include the action name you proposed")
    terminate: bool = Field(description="When the target has been achieved or the question has been answered, set this to True, otherwise False")

class AgentState(BaseModel):
    messages: Annotated[list, add_messages]
    action_description: str
    terminate: bool

class LangGraphAgentReact(LangGraphSupporter):
    def __init__(self,model:str,session_id:Optional[str] = "AgentSession" + get_date(),**kwargs):
        self.model = model
        self.agent_schema = ReACT
        self.graph_state = AgentState
        random_id = str(uuid.uuid4()) # important, other wise looped executor will have overlap memory and cause errors
        self.executor_agent = LangGraphAgent(model=self.model,session_id="ReACT_executor"+random_id)
        self.llm = ChatOpenAI(temperature=0, model=model, **kwargs)
        super().__init__(session_id=session_id,**kwargs)
        
    def switch_executor(self,executor):
        """Switch the executor agent
        Executor must be runnable and have an invoke method and clear_memory method

        Args:
            executor (_type_): _description_
        """
        self.executor_agent = executor
        self._create_agent()

    def _initialize_session_memory(self):
        system_message = """You run in a loop of Thought, Action, Terminate, Observation.
            At the end of the loop you output an Answer
            Use Thought to describe your thoughts about the question you have been asked.
            Use Action to run one of the actions available to you - then return PAUSE.
            Observation will be the result of running those actions.
            
            Example session:
            For example if you have a tool called search_wikipedia that takes a query and returns a result.
            
            Question: What is the capital of France?
            thought: I should look up France on Wikipedia
            action: perform a websearch what is the capital of France.
            Terminate: False 

            You will be called again with this:

            Observation: France is a country. The capital is Paris.
            Thought: The capital of France is Paris
            Action: No action is required
            Terminate: True 
            """
        return self.write_system_message(system_message)
    
    def _create_agent(self):
        self.structured_llm = self.llm.with_structured_output(self.agent_schema)
        
        self.executor_agent.add_tool(self.tools)
        workflow = StateGraph(AgentState)
        workflow.add_node("Reasoner", self._agent_thought_process)
        workflow.add_node("Executor", self._agent_action)
        workflow.add_edge(START, "Reasoner")
        workflow.add_conditional_edges("Reasoner", self._conditional_edge, {"continue": "Executor", END: END})
        workflow.add_edge("Executor", "Reasoner")
        self.agent = workflow.compile(checkpointer=self.memory_manager)

    def _agent_thought_process(self,state: AgentState):
        """Update memory and state

        Returns:
            _type_: _description_
        """
        # Retrieve the last message from the user
        parsed_response = self.structured_llm.invoke(state["messages"]).model_dump()
        # Append the AI's thoughts to the message history
        thought = parsed_response["thought"]
        action = parsed_response["action"]
        # format of <Thought>: thought, \n <Action>: action,\n <Terminate>: terminate
        combined_context = f"Thought: {thought}, \n Action: {action},\n Terminate: {parsed_response['terminate']}" 
        ai_message = AIMessage(content=combined_context)
        return {"messages":[ai_message],"action_description":action,"terminate":parsed_response["terminate"]}

    def _agent_action(self,state: AgentState):
        action_description = state["action_description"]
        output = self.executor_agent.invoke(action_description)
        # try key messages, if not, try chat_history #TODO: refactor this 
        if type(output)==list:
            observation = output[-1].content
        elif "output" in output.keys():
            observation = output["output"] #TODO: refer to intermediate steps, not just the output
            if output["intermediate_steps"] != []:
                tool_called = output["intermediate_steps"][0][0].tool
                observation = f"Tool called: {tool_called}, Output: {observation}"
        else:
            raise Exception("Output does not contain messages or chat_history")
        self.executor_agent.clear_memory()
        observation = f"Observation: {observation}"
        human_message = HumanMessage(content=observation)
        return {"messages":[human_message]}
    
    def _conditional_edge(self,state: AgentState):
        """Check if the user wants to continue the conversation

        Returns:
            _type_: _description_
        """
        return "continue" if state["terminate"]==False else END
    