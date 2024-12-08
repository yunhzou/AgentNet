from pydantic import BaseModel, Field, model_validator,ValidationError
from .langgraph_agent import LangGraphAgent
from typing import Optional, get_args 
# generate objective and output for the agent 
from .langgraph_agent import LangGraphAgent
from pydantic import BaseModel, Field
from typing import Optional, Annotated, TypedDict, Union,List
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
import uuid
from langchain_openai import ChatOpenAI
from .langgraph_supporter import LangGraphSupporter
from .agent_utils import get_date, extract_all_text_exclude_edges
from AgentNet.config import nosql_service

class LangGraphAgentCritic(LangGraphSupporter):
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
    class Critic(BaseModel):
        critic: str = Field(description="reasoning about if a task is successfully finished. This will support the judgment of success")
        success: bool = Field(description="If the you believe the task is completed, set to True, otherwise, False")
        
        
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
        critic: str
        success: bool

    def __init__(self, model: str, session_id: Optional[str] = "AgentSession" + get_date(), **kwargs):
        self.model = model
        self.agent_schema = self.Critic
        self.graph_state = self.AgentState
        self.llm = ChatOpenAI(temperature=0, model=model, **kwargs)
        super().__init__(session_id=session_id, **kwargs)

    def _initialize_session_memory(self):
        if self.system_message:
            return self.write_system_message(self.system_message)
        else:
            system_message = """
            You are a thoughtful critical agent. You help validate if a task is successfully finished.
            You will be given the context of a task and how it is solved.
            You reason if the approach is correct.
            At the end, you will judge if the task is successful or not.

            You should identify things such as error, wrong approach, or missing steps.

            Be cautious and thoughtful in your judgment.
            """
            return self.write_system_message(system_message)
    
    def _create_agent(self):
        self.structured_llm = self.llm.with_structured_output(self.agent_schema)
        workflow = StateGraph(self.graph_state)
        workflow.add_node("CriticAgent", self._agent_thought_process)
        workflow.add_edge(START, "CriticAgent")
        workflow.add_edge("CriticAgent", END)
        self.agent = workflow.compile(checkpointer=self.memory_manager)

    def _agent_thought_process(self,state: AgentState):
        """Update memory and state

        Returns:
            _type_: _description_
        """
        # Retrieve the last message from the user
        parsed_response = self.structured_llm.invoke(state["messages"]).model_dump()
        # Append the AI's thoughts to the message history
        critic = parsed_response["critic"]
        success = parsed_response["success"]
        response = critic + "Is the task successful? " + str(success)   
        response = AIMessage(content=response)
        return {"messages":[response],"critic":critic,"success":success}
