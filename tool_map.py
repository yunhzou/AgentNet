from langchain_core.tools import tool
import requests
from tavily import TavilyClient
from prefect import flow, task
from prefect.deployments import run_deployment


@tool
def addition(a:float,b:float)->float:
    """add two numbers"""
    return a+b

@tool
def multiplication(a:float,b:float)->float:
    """multiply two numbers"""
    return a*b

from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
python_repl = PythonREPL() 
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)


@tool
def read_webpage(url: str) -> str:
    """Read the webpage at the given URL, use whenever you need to access the content of a webpage. 

    Args:
        url (str): The URL of the webpage to read

    Returns:
        str: The content of the webpage
    """
    response = requests.get("https://r.jina.ai/" + url)
    return response.text


@tool
def web_search(query:str):
    """
    Search the web for information using Tavily, return urls and general information about the urls

    Args:
        query (str): The query to search for
    """
    tavily_client = TavilyClient(api_key= os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query, max_results=5)
    return response

from AgentNet.Agent import PerplexityChatBot

@tool
def perplexity_ai_search(query:str):
    """
    Search the web for information

    Args:
        query (str): The query to search for
    """
    session_id:str = "web_search"
    chatbot = PerplexityChatBot(model="llama-3.1-sonar-large-128k-online",session_id=session_id)
    response = chatbot.invoke(query)
    return response.content

import os
from langchain_core.tools import tool
from AgentNet.config import nosql_service

@tool
def Propose_New_Action(action_name:str, action_description:str):
    """
    Propose a new action, use this when action is not available but needed to solve the problem.
    action_name: str, the name of the action(tool)
    action_description: str, the description of the action(tool)
    """
    # insert the proposed action to the database
    nosql_service["ToolProposal"]["ToolProposal"].insert_one('proposed_actions', {'action_name': action_name, 'action_description': action_description})
    return f"action {action_name} is proposed"

tool_map={"addition":addition,
          "multiplication":multiplication,
          "web_search":web_search,
          "read_webpage":read_webpage,
          "python_repl":repl_tool,
          }
