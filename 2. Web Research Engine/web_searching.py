from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


def web_search(web_query: str, num_results: int) -> List[str]:
    return [r["link"]
        for r in DuckDuckGoSearchAPIWrapper().results(
            web_query, num_results)]


def web_scrape(url: str) -> str:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)

            return page_text
        else:
            return f"Could not retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Could not retrieve the webpage: {e}"
