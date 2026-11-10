import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

tavily = TavilyClient(api_key=api_key)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs and snippets."""
    try:
        results = tavily.search(query=query, max_results=5)
    except Exception as e:
        return f"Web search failed: {e}"

    out = []
    for r in results.get("results", []):
        snippet = (r.get("content") or "")[:300]
        out.append(
            f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {snippet}\n"
        )

    if not out:
        return "No results found."

    return "\n----\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:3000]
    except requests.RequestException as e:
        return f"Could not scrape URL: {e}"
