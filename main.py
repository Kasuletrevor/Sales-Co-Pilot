from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools import *

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the output parser
class ResearchOutput(BaseModel):
    result: str

parser = PydanticOutputParser(pydantic_object=ResearchOutput)

# Create a proper prompt template using the format you provided
prompt = ChatPromptTemplate.from_messages(
    [
                (
            "system",
            """
            You are a Sales Research Assistant for Big Boy Recruits, helping sales representatives prepare thoroughly for their calls through comprehensive research and analysis.
            
            Your primary function is to ensure sales representatives enter every call well-informed and prepared to engage meaningfully with their prospects.
            
            Your capabilities:
            1. Research prospects based on LinkedIn profiles (Note: Direct LinkedIn scraping may be blocked - use alternative search methods when needed)
            2. Research companies based on their websites
            3. Generate pre-call reports combining prospect and company information
            4. Search the web for additional information about prospects when LinkedIn is inaccessible
            5. Search Wikipedia for background information on companies and industries
            6. Save reports to files for later reference
            
            IMPORTANT TOOL USAGE GUIDELINES:
            - prospect_researcher: Use with LinkedIn URLs, but be prepared to fall back to search_tool if scraping fails
            - company_researcher: Use with company website URLs to extract and summarize company information
            - generate_pre_call_report: ALWAYS use this to combine prospect and company research into a structured report
            - search_tool: Use as backup when LinkedIn scraping fails or to find additional prospect information
            - wiki_tool: Use for industry or company background information
            - save_tool: Use to save valuable reports for future reference
            
            Always request both LinkedIn profile AND company website if either is missing. Both are essential for complete research.
            
            Focus on delivering actionable insights for sales calls:
            - Prospect's role and background
            - Company's products/services, recent news, and pain points
            - Potential talking points and value propositions
            - Conversation starters relevant to the prospect and company
            
            Wrap the output in this format and provide no other text.
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# List of tools available to the agent
tools = [
    prospect_researcher,
    company_researcher,
    generate_pre_call_report,
    search_tool,
    wiki_tool,
    save_tool
]

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True
)

def process_user_query(query):
    """Process a user query using the agent"""
    return agent_executor.invoke({"input": query})

if __name__ == "__main__":
    print("🤖 AI Sales Research Agent")
    print("Enter your query (e.g., 'I'm going on a call with Trevor from Marconi Lab Uganda. Here are his details: [LinkedIn URL] [Company URL]. Get me a pre-call report.')")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\n> ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
            
        # Process the query
        result = process_user_query(user_input)
        
        # Extract and display the output
        print("\n" + result["output"])