import os
from dotenv import load_dotenv

load_dotenv("keys.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or ""