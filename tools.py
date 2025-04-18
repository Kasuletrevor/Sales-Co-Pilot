import requests
import json
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from duckduckgo_search import DDGS
import wikipedia
from bs4 import BeautifulSoup

class LinkedInProfileInput(BaseModel):
    linkedin_url: str = Field(description="LinkedIn profile URL of the prospect")

class CompanyResearchInput(BaseModel):
    company_url: str = Field(description="URL of the company website")

class PreCallReportInput(BaseModel):
    prospect_summary: str = Field(description="Summary of the prospect")
    company_summary: str = Field(description="Summary of the company")


def scrape_website(url: str) -> str:
    """Scrape content from a website URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract text
        text = soup.get_text(separator=" ", strip=True)
        
        # Clean up text (remove extra whitespace)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit text length to avoid token limits
        return text[:15000]  # Limit to ~15k characters
    except Exception as e:
        return f"Error scraping website: {str(e)}"
    
@tool
def prospect_researcher(linkedin_url: str) -> str:
    """
    Research a prospect based on their LinkedIn profile URL.
    Scrapes the public profile and generates a summary with key insights.
    """
    # Scrape LinkedIn profile
    profile_text = scrape_website(linkedin_url)
    
    # If scraping failed or returned limited content, include an error message
    if profile_text.startswith("Error scraping website"):
        return json.dumps({
            "linkedin_url": linkedin_url,
            "error": profile_text,
            "success": False
        })
    
    # Use LLM to generate a summary from the scraped content
    summarization_prompt = f"""
    Can you please take this LinkedIn profile information and summarize this into a 300-word summary with details on the user's background, key career experience, current role and duration in role. This is a summary for a sales rep to prepare for a call, so make sure all the relevant details are there for this purpose, it should be an overview that will rapidly bring someone up to speed on a prospect they are about to get on call with.

    Make it read like a resume, with Name, location, follower count, current company and position all listed one after the other at the top, before you break into an overview section and then experience etc.

    Here is the information:
    {profile_text}
    """
    
    # You'd typically call your LLM here with the summarization_prompt
    # For example, using a helper function that accesses your ChatOpenAI instance
    # The code below assumes you have a summarize_with_llm function
    
    try:
        # Import OpenAI here to avoid circular imports if needed
        from langchain_openai import ChatOpenAI
        
        # Create a one-off LLM instance for this summarization
        summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        summary = summarizer.invoke(summarization_prompt).content
        
        # Format result
        result = {
            "linkedin_url": linkedin_url,
            "prospect_summary": summary,
            "success": True
        }
        
        return json.dumps(result)
    
    except Exception as e:
        # If LLM summarization fails, return the raw data with an error note
        return json.dumps({
            "linkedin_url": linkedin_url,
            "scraped_content": profile_text[:1000] + "..." if len(profile_text) > 1000 else profile_text,
            "error": f"Failed to summarize with LLM: {str(e)}",
            "success": False
        })


@tool
def company_researcher(company_url: str) -> str:
    """
    Research a company based on their website URL.
    Scrapes the company website and generates a summary relevant to sales.
    """
    # Scrape company website
    company_text = scrape_website(company_url)
    
    # If scraping failed or returned limited content, include an error message
    if company_text.startswith("Error scraping website"):
        return json.dumps({
            "company_url": company_url,
            "error": company_text,
            "success": False
        })
    
    # Use LLM to generate a summary from the scraped content
    summarization_prompt = f"""
    Please analyze this company website information and create a comprehensive summary for a sales representative. 
    The summary should include:
    
    1. Company name and location
    2. Industry and market position
    3. Key products or services offered
    4. Target customers or markets
    5. Company size and reach (if available)
    6. Recent news, developments, or initiatives
    7. Potential pain points or challenges the company might be facing
    8. Possible value propositions that would appeal to this company
    
    Focus on information that would be most valuable for a sales representative preparing for a call.
    
    Company website content:
    {company_text}
    """
    
    try:
        # Import OpenAI here to avoid circular imports if needed
        from langchain_openai import ChatOpenAI
        
        # Create a one-off LLM instance for this summarization
        summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        summary = summarizer.invoke(summarization_prompt).content
        
        # Format result
        result = {
            "company_url": company_url,
            "company_summary": summary,
            "success": True
        }
        
        return json.dumps(result)
    
    except Exception as e:
        # If LLM summarization fails, return the raw data with an error note
        return json.dumps({
            "company_url": company_url,
            "scraped_content": company_text[:1000] + "..." if len(company_text) > 1000 else company_text,
            "error": f"Failed to summarize with LLM: {str(e)}",
            "success": False
        })


@tool
def generate_pre_call_report(prospect_summary: str, company_summary: str) -> str:
    """
    Generate a pre-call report combining prospect and company research.
    Structures the information into an actionable format for sales reps.
    """
    # Format for saving
    result = {
        "prospect_summary": prospect_summary,
        "company_summary": company_summary,
        "report_status": "generated"
    }
    
    return json.dumps(result)


@tool
def search_tool(query: str) -> str:
    """Search the web for information about a query."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return json.dumps([result for result in results])

@tool
def wiki_tool(topic: str) -> str:
    """Search Wikipedia for information about a topic."""
    try:
        # Search for the page
        search_results = wikipedia.search(topic, results=1)
        if not search_results:
            return json.dumps({"error": f"No Wikipedia article found for '{topic}'"})
            
        # Get the page content
        page = wikipedia.page(search_results[0])
        
        result = {
            "title": page.title,
            "url": page.url,
            "summary": wikipedia.summary(page.title, sentences=5),
            "content_preview": page.content[:1000] + "..."
        }
        
        return json.dumps(result)
    except wikipedia.exceptions.DisambiguationError as e:
        return json.dumps({"error": f"Disambiguation error: {str(e)}"})
    except wikipedia.exceptions.PageError:
        return json.dumps({"error": f"No Wikipedia article found for '{topic}'"})
    except Exception as e:
        return json.dumps({"error": f"Error: {str(e)}"})

@tool
def save_tool(content: str, filename: str) -> str:
    """Save content to a file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return json.dumps({"success": True, "message": f"Content saved to {filename}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})