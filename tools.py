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