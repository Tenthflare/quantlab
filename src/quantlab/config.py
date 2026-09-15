import os

from dotenv import load_dotenv

load_dotenv()

NASDAQ_API_KEY = os.environ.get("NASDAQ_DATA_LINK_API_KEY")
if not NASDAQ_API_KEY:
    raise RuntimeError("NASDAQ_DATA_LINK_API_KEY not set — create a .env file (see .env.example)")
