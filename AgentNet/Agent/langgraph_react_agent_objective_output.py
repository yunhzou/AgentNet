from pydantic import BaseModel, Field, model_validator,ValidationError
from .langgraph_agent import LangGraphAgent
from typing import Optional, get_args,Callable
# generate objective and output for the agent 
from .langgraph_agent import LangGraphAgent
from pydantic import BaseModel, Field
from typing import Optional, Annotated, TypedDict, Union,List
from langgraph.graph.message import add_messages
from operator import add
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END, START
import uuid
from langchain_openai import ChatOpenAI
from .langgraph_supporter import LangGraphSupporter
from .agent_utils import get_date, extract_all_text_exclude_edges
from AgentNet.config import nosql_service
from .langgraph_critic_agent import LangGraphAgentCritic

class LangGraphAgentReactExperimental(LangGraphSupporter):
    """
    Reasoner: Structured ReAct
    Executor: Simple action without reasoning, memory is cleared after each step.

    Args:
        LangGraphSupporter (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    class Action(BaseModel):
        objective: str = Field(description="Abstract Objective of the action's goal in one sentence. For example, the objective of action of 3+4 would be Perform a mathematical addition calculation")
        input_context: str = Field(description="Input context of the action in one sentence")
        output: str = Field(description="Description of the output of the action in one sentence. You must describe the abstraction of the output instead of the actual output. For example, the abstraction of the output of 3+4 would be a computed numerical value")

    class NoAction(BaseModel):
        message: str = Field(description="Explain why no action is required. For example, when you have the result to answer the original question, you can set this to 'objective achieved'.1")

    class ReACT_Experimental(BaseModel):
        thought: str = Field(description="One thought process. Each thought is only one sentence and should have an objective, you must follow this rule")
        action: Union["LangGraphAgentReactExperimental.Action", "LangGraphAgentReactExperimental.NoAction"] = Field(
            description="An action should be taken to achieve the thought. Abstract the action to an objective and output, and provide the detailed input context"
        )
        terminate: bool = Field(description="When the target has been achieved or the question has been answered, set this to True, otherwise False")

        @model_validator(mode='before')
        def validate_and_fix_structure(cls, values):
            """
            Validate and fix structure by normalizing field names and handling incorrect nesting.
            """
            def find_value(data, key_to_find):
                if isinstance(data, dict):
                    for key, value in data.items():
                        if key.lower() == key_to_find.lower():
                            return value
                        else:
                            found = find_value(value, key_to_find)
                            if found is not None:
                                return found
                elif isinstance(data, list):
                    for item in data:
                        found = find_value(item, key_to_find)
                        if found is not None:
                            return found
                return None

            def fix_data(data, model):
                if not isinstance(data, dict):
                    return data
                fixed_data = {}
                for field_name, field_info in model.model_fields.items():
                    field_value = None
                    # Try to get the value matching the field name, case-insensitive
                    for key, val in data.items():
                        if key.lower() == field_name.lower():
                            field_value = val
                            break
                    # If not found, try to find it in nested data
                    if field_value is None:
                        field_value = find_value(data, field_name)
                    # If field is a nested model or Union
                    if field_value is not None:
                        if hasattr(field_info.annotation, '__pydantic_model__'):
                            field_value = fix_data(field_value, field_info.annotation)
                        elif (
                            hasattr(field_info.annotation, '__origin__')
                            and field_info.annotation.__origin__ is Union
                        ):
                            # Try each type in the Union
                            for sub_type in get_args(field_info.annotation):
                                try:
                                    field_value_fixed = fix_data(field_value, sub_type)
                                    # Validate by trying to instantiate the sub_type
                                    sub_instance = sub_type(**field_value_fixed)
                                    field_value = field_value_fixed  # Valid sub_type found
                                    break
                                except ValidationError:
                                    continue
                    fixed_data[field_name] = field_value
                return fixed_data

            values = fix_data(values, cls)
            return values


    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        action_description: str 
        terminate: bool 
        formatted_history: Annotated[list, add]
        """Formatted history of the agent's thought, action_proposal,tool_usage, tool_input, observation, critic, and success"""


    def __init__(self, model: str, session_id: Optional[str] = "AgentSession" + get_date(), **kwargs):
        self.model = model
        self.agent_schema = self.ReACT_Experimental
        self.graph_state = self.AgentState
        random_id = str(uuid.uuid4())  # important, otherwise looped executor will have overlap memory and cause errors
        self.executor_agent = LangGraphAgent(model=self.model, session_id=session_id+"_Executor")
        self.critic_agent = LangGraphAgentCritic(model=self.model, session_id=session_id+"_Critic")
        self.llm = ChatOpenAI(temperature=0, model=model, **kwargs)
        super().__init__(session_id=session_id, **kwargs)
        self.formatted_history_record = None
        
    def switch_executor(self,executor):
        """Switch the executor agent
        Executor must be runnable and have an invoke method and clear_memory method

        Args:
            executor (_type_): _description_
        """
        self.executor_agent = executor
        self._create_agent()

    def _initialize_system_message(self):
        if self.system_message:
            return self.rewrite_system_message(self.system_message)
        else:
            system_message = """
                You run in a loop of Thought, Action, Terminate, Observation.
                Use Thought to describe your thoughts about the question you have been asked.
                Use Action to PROPOSE one of the actions available to you (Not actually ran at this step, must be one action at a time)
                Human will take your action proposal and run it. Observation will be given with the result of running those actions.
                You action can either be a tool or calling up another agent to perform the task.
                If its agent, you should properly route the task to the agents within your action proposal.

                Example session:
                For example if you have a tool called search_wikipedia that takes a query and returns a result.

                Question: What is the capital of France?
                thought: I should look up France on Wikipedia
                action: perform a websearch what is the capital of France with search_wikipedia.
                Terminate: False 

                You will be called again with this after human ran the action:

                Observation: France is a country. The capital is Paris.
                Thought: The capital of France is Paris
                Action: No action is required
                Terminate: True 

                If you see error or situation that you believe to be corrupted in observation, you should set terminate to True and describe the error in the observation
                Example session:
                Question: What is the capital of France?
                thought: I should look up France on Wikipedia
                action: perform a websearch what is the capital of France.
                Terminate: True
                Observation: Wikipedia is not available
                thought: the wikipedia is not available at the moment, and thus i cannot answer the question
                Action: No action is required
                Terminate: True
                """
            return self.rewrite_system_message(system_message)
    
    def _create_agent(self):
        self.structured_llm = self.llm.with_structured_output(self.agent_schema)
        
        self.executor_agent.add_tool(self.tools)
        workflow = StateGraph(self.graph_state)
        workflow.add_node("Reason", self._run_reasoner)
        workflow.add_node("Execute", self._run_executor)
        workflow.add_node("Critic", self._run_critic_agent)
        workflow.add_edge(START, "Reason")
        workflow.add_conditional_edges("Reason", self._conditional_edge, {"continue": "Execute", END: END})
        workflow.add_edge("Execute", "Critic")
        workflow.add_edge("Critic", "Reason")
        self.agent = workflow.compile(checkpointer=self.memory_manager)

    def _run_reasoner(self,state: AgentState):
        """Update memory and state

        Returns:
            _type_: _description_
        """
        # Retrieve the last message from the user
        parsed_response = self.structured_llm.invoke(state["messages"]).model_dump()
        # Append the AI's thoughts to the message history
        thought = parsed_response["thought"]
        action = parsed_response["action"]
        terminate = parsed_response["terminate"]
        # action is a dict, so we need to format it to a string with the format of <key>:<value>, <key2>:<value2>
        action = ", ".join([f"{key}: {value}" for key,value in action.items()])
        # format of <Thought>: thought, \n <Action>: action,\n <Terminate>: terminate
        combined_context = f"Thought: {thought}, \n Action: {action},\n Terminate: {parsed_response['terminate']}" 
        ai_message = AIMessage(content=combined_context)
        
        self.formatted_history_record = {"thought":thought,"action":action,"terminate":terminate} 
        
        return {"messages":[ai_message],"action_description":action,"terminate":terminate}

    def _run_executor(self,state: AgentState):
        """
        Executor acquire the task, and then select the tool to finish the task.

        """
        past_memory = state["messages"]
        # merge content
        merged_content = extract_all_text_exclude_edges(past_memory,last_num_messages=1)
        action_description = state["action_description"]
        context = f"CONTEXT: {merged_content} \n YOUR TASK:\n {action_description}"
        output = self.executor_agent.invoke_return_graph_state(context)
        """Adding tool information to the Observation"""
        tool_used = output["tool_used"]
        if len(tool_used) > 1:
            Warning("Multiple tools are called, could be a potential issue")
        tool_used = tool_used[0]
        observation = output["messages"][-1].content
        self.formatted_history_record["tool_used"] = tool_used
        self.formatted_history_record["observation"] = observation
        self.executor_agent.clear_memory(print_log=False)
    
    
    def _run_critic_agent(self,state:AgentState):
        """criticize task_response pair to check if the task is successful or not"""
        critic_content = ", ".join([f"{key}: {value}" for key,value in self.formatted_history_record.items()])
        critic_result = self.critic_agent.invoke_return_graph_state(critic_content)
        critic_reasoning = critic_result["critic"]
        success_judgement = critic_result["success"]
        self.formatted_history_record["critic"] = critic_reasoning
        self.formatted_history_record["success"] = success_judgement
        critic_message = f"Critic: {critic_reasoning}, Success?: {success_judgement}" + "\n"
        tool_used = self.formatted_history_record["tool_used"]
        tool_name = tool_used["tool_name"]
        tool_args = tool_used["args"]
        observation = self.formatted_history_record["observation"]
        observation_message = "Observation"+"\n"+f"Tool used: {tool_name}, Args: {tool_args}, Output: {observation}" + "\n"
        observation_critic_feedback_grounding = observation_message + critic_message
        human_message = self._strcture_user_input(observation_critic_feedback_grounding)
        return {"messages":[human_message],"formatted_history":[self.formatted_history_record]}
    

    def _conditional_edge(self,state: AgentState):
        """Check if the agent wants to continue the conversation

        Returns:
            _type_: _description_
        """
        return "continue" if state["terminate"]==False else END
    
    def inject_text_block_human_message(self,
                                        block_name:str,
                                        block_description:str, 
                                        block_update_function:Callable):
        """ Add a switchable text block to the agent memory"""
        block = (block_name, block_description, block_update_function)
        self.message_blocks.append(block)
        self.critic_agent.message_blocks.append(block)