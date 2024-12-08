from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from AgentNet.config import openai_api_key, mongodb_connection_string


class LangChainChatBot():
    def __init__(self,
                 model:str,
                 session_id: Optional[str] = "ChatSession",
                 **kwargs
                 ):
        """
        Chat bot with memory

        Args:
            client (_type_): OpenAI client
            model (_type_): choose from gpt-3.5-turbo-1106 or gpt-4-1106-preview
            usersetup (_type_): user instructions that the assitant will always follow
        """
        self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model=model, **kwargs)
        self.memory = MongoDBChatMessageHistory(connection_string=mongodb_connection_string,session_id=session_id)
        llm_chain = self._get_llm_chain()
        self.chat_chain = RunnableWithMessageHistory(
            llm_chain,
            lambda _: self.memory, #wierd here have to be a callable function just don't make sense
            input_messages_key="question",
            history_messages_key="history",
        )
        self.config = {"configurable": {"session_id": session_id}}

    def _get_llm_chain(self):
        prompt = self._get_prompt_template()
        llm_chain = prompt | self.llm
        return llm_chain

    def _get_prompt_template(self):
        
        #TODO: modify ChatPromptTemplate to add vision/image mode, refer vision.py \ 
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You're an assistant"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        return prompt
        
    
    def invoke(self,prompt):
        response = self.chat_chain.invoke(
            {"question": prompt},
             config = self.config
        )
        print(response.content)
        return response
    
    def clear_memory(self):
        self.memory.clear()



class O1ChatBot(LangChainChatBot):
    def __init__(self,
                 model:str,
                 session_id: Optional[str] = "ChatSession",
                 **kwargs
                 ):
        """
        Chat bot with memory

        Args:
            client (_type_): OpenAI client
            model (_type_): choose from gpt-3.5-turbo-1106 or gpt-4-1106-preview
            usersetup (_type_): user instructions that the assitant will always follow
        """
        self.llm = ChatOpenAI(temperature=1, openai_api_key=openai_api_key, model=model, **kwargs)
        self.memory = MongoDBChatMessageHistory(connection_string=mongodb_connection_string,session_id=session_id)
        llm_chain = self._get_llm_chain()
        self.chat_chain = RunnableWithMessageHistory(
            llm_chain,
            lambda _: self.memory, #wierd here have to be a callable function just don't make sense
            input_messages_key="question",
            history_messages_key="history",
        )
        self.config = {"configurable": {"session_id": session_id}}
    def _get_prompt_template(self):
        
        #TODO: modify ChatPromptTemplate to add vision/image mode, refer vision.py \ 
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        return prompt