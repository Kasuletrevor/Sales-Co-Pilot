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
    
    # Format for saving (optional)
    result = {
        "linkedin_url": linkedin_url,
        "scraped_content": profile_text[:1000] + "...",  # Truncated for readability
        "full_content_length": len(profile_text)
    }
    
    return json.dumps(result)


@tool
def company_researcher(company_url: str) -> str:
    """
    Research a company based on their website URL.
    Scrapes the company website and generates a summary relevant to sales.
    """
    # Scrape company website
    company_text = scrape_website(company_url)
    
    # Format for saving (optional)
    result = {
        "company_url": company_url,
        "scraped_content": company_text[:1000] + "...",  # Truncated for readability
        "full_content_length": len(company_text)
    }
    
    return json.dumps(result)


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