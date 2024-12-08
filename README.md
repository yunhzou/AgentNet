# AgentNet
Following cognitive architecture of language agents, AgentNet provide implementation strategy that is scalable for procedure, semantic and episodic memory.  
  
Agent Net features a tree of agents to form logical decision trees. The lowest level agent always associated with tool/action, while the higher level agent serves as routers and decision making unit to plan sub task. So a major task is first disected to subgoals and procedures, and finally to executable actions. 

# Use Case
El-Agente: Computational Chemists    
General Embedded Agents.  

# Installation:   
run:  
pip install -r setup\requirements.txt  

# Setup:
Create .env file, you dont need LANGCHAIN if you don't use langsmith to track the log. Its recommended that you include them. 
OPENAI_API_KEY="<your-openai-api-key>"  
MONGODB_URL="<your-mongodb-url>"  
LANGCHAIN_TRACING_V2=<true-or-false>  
LANGCHAIN_ENDPOINT="<langchain-endpoint>"  
LANGCHAIN_API_KEY="<your-langchain-api-key>"  
LANGCHAIN_PROJECT="<your-langchain-project-name>"  
PPLX_API_KEY="<your-pplx-api>"  
TAVILY_API_KEY="<your-tavily-api-key>"  
SERPER_API_KEY="<your-serper-api-key>"  