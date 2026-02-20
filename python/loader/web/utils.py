import httpx
from html.parser import HTMLParser
from urllib.parse import urlparse

def detect_url_type(url: str) -> str:
    host = urlparse(url).netloc.lower().lstrip("www.")
    if host == "github.com":
        return "github"
    if host in ("youtube.com", "youtu.be", "m.youtube.com"):
        return "youtube"
    return "web"


def get(url: str, *, timeout: int = 30, headers: dict | None = None) -> httpx.Response:
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Spider-Bot/2.0; "
            "+https://github.com/nathsujal/spider)"
        )
    }
    if headers: 
        default_headers.update(headers)
    resp = httpx.get(url, follow_redirects=True, timeout=timeout, headers=default_headers)
    resp.raise_for_status()
    return resp