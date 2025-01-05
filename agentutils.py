import re
from bs4 import BeautifulSoup
import requests
import spacy


nlp = spacy.load("en_core_web_sm")

def research(query):
    """
    Placeholder for a more sophisticated web research function.
    For now, it performs a simple Google search and returns the text content of the first result.
    """
    try:
        search_url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}  # Basic user agent to avoid being blocked
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, "html.parser")
        first_result_link = soup.find("a", href=re.compile(r"https://"))

        if first_result_link:
            result_url = first_result_link["href"]
            result_response = requests.get(result_url, headers=headers)
            result_response.raise_for_status()

            result_soup = BeautifulSoup(result_response.content, "html.parser")
            paragraphs = result_soup.find_all("p")
            text_content = " ".join([p.get_text() for p in paragraphs])
            return text_content[:5000] # Limit text length

        return "No results found."

    except requests.exceptions.RequestException as e:
        print(f"Error during web research: {e}")
        return "Error during research."

def analyze_data(data):
    """
    Placeholder for data analysis using spaCy.
    """
    doc = nlp(data)
    # Example: Extract key entities
    key_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]]
    return {"key_entities": key_entities}

