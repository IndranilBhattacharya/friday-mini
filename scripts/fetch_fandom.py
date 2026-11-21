"""
Download article text from the Marvel Fandom wikis (CC BY-SA licensed, like Wikipedia). Resumable.
  python scripts/fetch_fandom.py
Output: data/raw/pretrain/fandom_<wiki>.txt (one article per line). Credit Fandom contributors in the model card.
"""
import os, random, re, time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import requests

HEADERS = {"User-Agent": "slm-learning-project/0.1 (educational)"}
WIKIS = ["marvelcinematicuniverse", "marvel"]       # Marvel Cinematic Universe Wiki, Marvel Database (comics)
MAX_PAGES = 40000                                   # per wiki (random sample of all main-namespace pages)


def api(wiki, params):
    for wait in (1, 5, 20, 60, 120):                # polite retry
        try:
            r = requests.get(f"https://{wiki}.fandom.com/api.php", params={**params, "format": "json"}, headers=HEADERS, timeout=60)
            return r.json()
        except (ValueError, requests.RequestException):
            time.sleep(wait)
    raise RuntimeError("fandom keeps failing")


class Paragraphs(HTMLParser):
    """Keeps only the text inside <p> tags (skips infoboxes, tables, navigation, scripts)."""

    def __init__(self):
        super().__init__()
        self.depth, self.skip, self.out = 0, 0, []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "table", "aside", "sup"):
            self.skip += 1
        elif tag == "p" and not self.skip:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "table", "aside", "sup"):
            self.skip = max(0, self.skip - 1)
        elif tag == "p" and self.depth:
            self.depth -= 1
            self.out.append(" ")

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.out.append(data)


def article_text(wiki, title):
    d = api(wiki, {"action": "parse", "page": title, "prop": "text", "disablelimitreport": 1, "redirects": 1})
    html = d.get("parse", {}).get("text", {}).get("*", "")
    p = Paragraphs()
    p.feed(html)
    return " ".join("".join(p.out).split())


def list_titles(wiki):
    titles, cont = [], {}
    while len(titles) < 150000:
        d = api(wiki, {"action": "query", "list": "allpages", "aplimit": 500, "apfilterredir": "nonredirects", **cont})
        titles += [p["title"] for p in d["query"]["allpages"] if "/" not in p["title"]]   # skip /Gallery, /Quotes ...
        if "continue" not in d:
            break
        cont = {"apcontinue": d["continue"]["apcontinue"]}
        time.sleep(0.3)
    random.Random(0).shuffle(titles)                # same sample every run => resumable
    return titles[:MAX_PAGES]


def job(args):
    wiki, title = args
    try:
        time.sleep(0.4)
        return title, article_text(wiki, title)
    except Exception:
        return title, None


if __name__ == "__main__":
    for wiki in WIKIS:
        done_path, out_path = f"data/fandom_{wiki}.done", f"data/raw/pretrain/fandom_{wiki}.txt"
        done = set(open(done_path, encoding="utf-8").read().splitlines()) if os.path.exists(done_path) else set()
        todo = [(wiki, t) for t in list_titles(wiki) if t not in done]
        print(wiki, len(todo), "to download", flush=True)
        with open(out_path, "a", encoding="utf-8") as out, open(done_path, "a", encoding="utf-8") as dn, ThreadPoolExecutor(3) as ex:
            for i, (title, text) in enumerate(ex.map(job, todo)):
                if text is None:
                    continue
                if len(text) > 500:
                    out.write(text + "\n"); out.flush()
                dn.write(title + "\n"); dn.flush()
                if i % 200 == 0:
                    print(wiki, i, title, flush=True)
