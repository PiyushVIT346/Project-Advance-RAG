import re

def contains_keywords(text, patterns):
    return any(re.search(p, text.lower()) for p in patterns)
