import requests
import json
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from duckduckgo_search import DDGS
import wikipedia
from bs4 import BeautifulSoup