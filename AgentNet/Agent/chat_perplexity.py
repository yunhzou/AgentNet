from langchain_community.chat_models import ChatPerplexity
from .langchain_chatbot import LangChainChatBot
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from AgentNet.config import mongodb_connection_string
from typing import Optional

class PerplexityChatBot(LangChainChatBot):
    def __init__(self,
                model:str,
                session_id: Optional[str] = "ChatSession",
                **kwargs
                ):
        """
        Chat bot with memory

        Args:
            client (_type_): OpenAI client
            model (_type_): choose from perplexity models
            usersetup (_type_): user instructions that the assitant will always follow
        """
        self.llm = ChatPerplexity(temperature=0, model=model, **kwargs)
        self.memory = MongoDBChatMessageHistory(connection_string=mongodb_connection_string,session_id=session_id)
        llm_chain = self._get_llm_chain()
        self.chat_chain = RunnableWithMessageHistory(
            llm_chain,
            lambda _: self.memory, #wierd here have to be a callable function just don't make sense
            input_messages_key="question",
            history_messages_key="history",
        )
        self.config = {"configurable": {"session_id": session_id}}