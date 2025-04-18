from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
# from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# response = llm.invoke("What is the capital of France?")
# print(response)

# Define the agent's system prompt
system_prompt = """
You are a Sales Research Assistant that helps sales representatives prepare for sales calls by researching prospects and companies.

Your capabilities:
1. Research prospects based on their LinkedIn profiles 
2. Research companies based on their websites
3. Generate pre-call reports combining prospect and company information
4. Search the web for additional information
5. Search Wikipedia for background information
6. Save reports to files for later reference

When asked to research a prospect or company, always use the appropriate tools.
Provide concise, sales-focused insights that would be valuable for a sales representative.
"""