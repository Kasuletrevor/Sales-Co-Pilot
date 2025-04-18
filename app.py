import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import json
import time
import uuid  # Add this import for generating unique IDs
from tools import *

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Sales Co-Pilot",
    page_icon="🤝",
    layout="wide",
)

# Initialize the app state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "agent_executor" not in st.session_state:
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Define the output parser
    class ResearchOutput(BaseModel):
        result: str
    
    parser = PydanticOutputParser(pydantic_object=ResearchOutput)
    
    # Create the prompt template
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
    st.session_state.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True
    )

# App header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🤝 Sales Co-Pilot")
    st.markdown("### Research Assistant for Sales Representatives")
with col2:
    st.image("https://img.icons8.com/color/96/000000/sales-performance.png", width=100)

# App description
with st.expander("About this app", expanded=False):
    st.markdown("""
    **Sales Co-Pilot** helps you prepare for sales calls by:
    
    1. **Researching prospects** from LinkedIn profiles
    2. **Analyzing companies** from their websites
    3. **Generating pre-call reports** with actionable insights
    
    Simply enter your request with the LinkedIn URL of the prospect and the company website URL.
    
    Example: "I'm meeting John Smith from Acme Corp tomorrow. His LinkedIn is www.linkedin.com/in/john-smith and the company website is https://acmecorp.com. Can you prepare a pre-call report?"
    """)

# Create a container for chat history
chat_container = st.container()

# User input area
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_area("Enter your request:", 
                              placeholder="Example: I'm meeting John Smith from Acme Corp tomorrow. His LinkedIn is www.linkedin.com/in/john-smith and the company website is https://acmecorp.com. Can you prepare a pre-call report?",
                              height=100)
    col1, col2 = st.columns([4, 1])
    with col2:
        submit_button = st.form_submit_button("🔍 Research")

# Process the user's input when the form is submitted
if submit_button and user_input:
    # Show a spinner while processing
    with st.spinner("Researching... This might take a minute"):
        # Generate a unique ID for this message
        message_id = str(uuid.uuid4())
        
        # Add user message to chat history with unique ID
        st.session_state.chat_history.append({
            "id": message_id,
            "role": "user", 
            "content": user_input,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Process the query with the agent
        result = st.session_state.agent_executor.invoke({"input": user_input})
        
        # Extract and display the output
        try:
            # Try to parse as JSON for better formatting
            output_data = json.loads(result["output"])
            
            # Different handling based on what's in the output
            if "pre_call_report" in output_data:
                output_content = output_data["pre_call_report"]
            elif "result" in output_data:
                output_content = output_data["result"]
            else:
                output_content = result["output"]
                
            # Add assistant message to chat history with unique ID
            st.session_state.chat_history.append({
                "id": str(uuid.uuid4()),
                "role": "assistant", 
                "content": output_content,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, use the raw output
            st.session_state.chat_history.append({
                "id": str(uuid.uuid4()),
                "role": "assistant", 
                "content": result["output"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

# Display chat history
with chat_container:
    for message in reversed(st.session_state.chat_history):  # Display newest messages first
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(f"**{message.get('timestamp', '')}**")
                st.markdown(message['content'])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(f"**{message.get('timestamp', '')}**")
                
                # For assistant messages, create an expander with the content
                with st.expander("**Sales Co-Pilot Report**", expanded=True):
                    # Display the markdown content
                    st.markdown(message['content'])
                    
                    # Create a unique key using the message's ID
                    save_key = f"save_{message['id']}"
                    
                    # Add a download button for the report with a unique key
                    if st.button("💾 Save Report", key=save_key):
                        filename = f"sales_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(message["content"])
                        st.success(f"Report saved to {filename}")

# Sidebar with options
with st.sidebar:
    st.title("Options")
    
    # Clear chat history button
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    # Model selection
    st.subheader("Model Settings")
    model = st.selectbox(
        "Select AI Model",
        ["gpt-4o-mini", "gpt-4"],
        index=0
    )
    
    # Example prompts
    st.subheader("Example Prompts")
    example_prompts = [
        "I have a call with Sarah Johnson from TechCorp tomorrow. Her LinkedIn is www.linkedin.com/in/sarah-johnson and company website is https://techcorp.com",
        "Research this prospect: www.linkedin.com/in/john-doe and their company: https://example.org",
        "I need information about Acme Corp (https://acme.com) for my sales call"
    ]
    
    for i, example in enumerate(example_prompts):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            # Set the form value through session state
            st.session_state["user_input"] = example
            st.experimental_rerun()
    
    # App info
    st.markdown("---")
    st.caption("Sales Co-Pilot v1.0")
    st.caption("© 2025 Big Boy Recruits")