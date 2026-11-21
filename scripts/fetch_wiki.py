"""
Download Marvel-related Wikipedia article text (free, CC BY-SA) into data/raw/pretrain/wiki.txt, one article per line.
  python scripts/fetch_wiki.py
Credit Wikipedia in the model card if you publish.
"""
import os, time, requests
from concurrent.futures import ThreadPoolExecutor

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "slm-learning-project/0.1 (educational)"}
CATEGORIES = [
    "Category:Marvel Cinematic Universe films", "Category:Marvel Cinematic Universe television series",
    "Category:Marvel Comics superheroes", "Category:Marvel Comics supervillains",
    "Category:Marvel Comics titles", "Category:Marvel Cinematic Universe characters",
    "Category:Marvel Comics superhero teams and groups", "Category:Marvel Comics characters with superhuman strength",
    "Category:Marvel Comics mutants", "Category:Marvel Comics deities", "Category:Marvel Comics extraterrestrial superheroes",
    "Category:Marvel Comics film characters", "Category:Marvel Comics limited series", "Category:Marvel Comics storylines",
]


def get(params):
    for wait in (1, 5, 20, 60, 120):                 # polite retry: Wikipedia sometimes answers empty when we go too fast
        r = requests.get(API, params={**params, "format": "json"}, headers=HEADERS, timeout=60)
        try:
            return r.json()
        except ValueError:
            time.sleep(wait)
    raise RuntimeError("Wikipedia keeps failing: " + r.text[:200])


def category_titles(cat):
    titles, cont = [], {}
    while True:
        d = get({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": 500, "cmtype": "page", **cont})
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return titles
        cont = d["continue"]


def extract(title):
    """Full plain text of ONE article (the API only gives one full extract per request)."""
    d = get({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1})
    return next(iter(d["query"]["pages"].values())).get("extract", "")


def safe_extract(title):
    try:
        time.sleep(0.5)                       # be polite: Wikipedia rate-limits fast clients
        return title, extract(title)
    except Exception:
        return title, None                    # skipped now, retried on the next run


if __name__ == "__main__":
    os.makedirs("data/raw/pretrain", exist_ok=True)
    titles = []
    for c in CATEGORIES:
        titles += category_titles(c)
    titles = sorted(set(titles))
    done = set(open("data/wiki.done", encoding="utf-8").read().splitlines()) if os.path.exists("data/wiki.done") else set()
    todo = [t for t in titles if t not in done]               # resumable: finished titles are remembered
    print(len(titles), "articles,", len(todo), "to download", flush=True)
    with open("data/raw/pretrain/wiki.txt", "a", encoding="utf-8") as f, open("data/wiki.done", "a", encoding="utf-8") as d:
        with ThreadPoolExecutor(3) as ex:
            for i, (title, text) in enumerate(ex.map(safe_extract, todo)):
                if text is None:
                    continue
                text = " ".join(text.split())
                if len(text) > 500:
                    f.write(text + "\n"); f.flush()
                d.write(title + "\n"); d.flush()
                if i % 100 == 0:
                    print(i, title, flush=True)
