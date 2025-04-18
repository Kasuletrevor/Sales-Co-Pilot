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
llm = ChatOpenAI(model="gpt-4", temperature=0)

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