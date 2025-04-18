from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import *

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

# List of tools available to the agent
tools = [
    prospect_researcher,
    company_researcher, 
    generate_pre_call_report,
    search_tool,
    wiki_tool,
    save_tool
]

# Create the agent
agent = create_tool_calling_agent(llm, tools, system_prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def research_prospect(linkedin_url):
    """Research a prospect based on their LinkedIn URL"""
    return agent_executor.invoke({
        "input": f"Research this prospect with LinkedIn URL: {linkedin_url}"
    })

def research_company(company_url):
    """Research a company based on their website URL"""
    return agent_executor.invoke({
        "input": f"Research this company with URL: {company_url}"
    })

def generate_report(prospect_data, company_data):
    """Generate a pre-call report based on prospect and company research"""
    return agent_executor.invoke({
        "input": f"Generate a pre-call report for this prospect: {prospect_data} and company: {company_data}"
    })


if __name__ == "__main__":
    print("🤖 AI Sales Research Agent")
    print("1. Research Prospect")
    print("2. Research Company")
    print("3. Generate Pre-Call Report")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        linkedin_url = input("Enter LinkedIn URL: ")
        result = research_prospect(linkedin_url)
        print(result["output"])
        
    elif choice == "2":
        company_url = input("Enter company website URL: ")
        result = research_company(company_url)
        print(result["output"])
        
    elif choice == "3":
        prospect_summary = input("Enter prospect summary or paste from previous research: ")
        company_summary = input("Enter company summary or paste from previous research: ")
        result = generate_report(prospect_summary, company_summary)
        print(result["output"])
        
    else:
        print("Invalid choice. Please run the program again.")