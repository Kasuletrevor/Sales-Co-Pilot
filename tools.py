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