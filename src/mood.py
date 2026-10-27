import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommender import load_bundle

try:
    sys.stdout.reconfigure(encoding="utf-8")   # lets Windows print emojis
except Exception:
    pass

# mood -> (emojis, synonyms, genre keywords to match in 'listed_in', query words)
MOODS = {
    "happy": {
        "emojis": ["😂", "😄", "😊", "😁", "🤣"],
        "words": ["happy", "funny", "laugh", "fun", "cheerful", "joy", "silly", "lighthearted"],
        "genres": ["comedies", "comedy", "family", "kids", "stand-up", "music"],
        "query": "funny hilarious laugh lighthearted fun friends humor",
    },
    "sad": {
        "emojis": ["😢", "😭", "😞", "🥺", "💔"],
        "words": ["sad", "cry", "down", "lonely", "heartbroken", "emotional", "blue"],
        "genres": ["dramas", "drama", "romantic", "independent"],
        "query": "emotional touching loss heartfelt moving grief family struggle",
    },
    "scared": {
        "emojis": ["😱", "😨", "👻", "🎃", "🔪"],
        "words": ["scared", "fear", "horror", "spooky", "creepy", "scary", "thrill"],
        "genres": ["horror", "thrillers", "thriller", "mysteries"],
        "query": "terrifying haunted dark evil supernatural killer fear night",
    },
    "romantic": {
        "emojis": ["😍", "🥰", "❤️", "💕", "😘"],
        "words": ["romantic", "love", "romance", "date", "crush", "cute"],
        "genres": ["romantic", "romance"],
        "query": "love romance relationship couple falls in love wedding heart",
    },
    "excited": {
        "emojis": ["🤩", "🔥", "💥", "😎", "🚀"],
        "words": ["excited", "action", "adrenaline", "pumped", "energetic", "thrilling", "epic"],
        "genres": ["action", "adventure", "sci-fi", "thrillers"],
        "query": "action battle fight explosive chase mission hero adventure danger",
    },
    "relaxed": {
        "emojis": ["😌", "😴", "☕", "🛋️", "🌿"],
        "words": ["relaxed", "calm", "chill", "cozy", "peaceful", "unwind", "tired", "sleepy"],
        "genres": ["docuseries", "nature", "reality", "stand-up", "cooking", "lifestyle"],
        "query": "relaxing calm gentle nature cooking travel everyday life easygoing",
    },
    "curious": {
        "emojis": ["🤔", "🧐", "🔍", "💡", "🧠"],
        "words": ["curious", "learn", "think", "documentary", "smart", "interesting", "mystery"],
        "genres": ["documentaries", "documentary", "docuseries", "science", "mysteries", "crime"],
        "query": "true story investigation discovery world history science explores truth",
    },
    "adventurous": {
        "emojis": ["🌍", "🧭", "🏔️", "🗺️", "🦸"],
        "words": ["adventurous", "adventure", "explore", "fantasy", "magic", "journey"],
        "genres": ["adventure", "fantasy", "sci-fi", "anime"],
        "query": "journey quest magical world explore kingdom creatures discover",
    },
    "angry": {
        "emojis": ["😡", "😤", "🤬", "👊"],
        "words": ["angry", "mad", "furious", "revenge", "frustrated", "rage"],
        "genres": ["action", "crime", "thrillers"],
        "query": "revenge fight justice vengeance rage corrupt gangster battle",
    },
    "family": {
        "emojis": ["👨‍👩‍👧", "👶", "🧸", "🎈"],
        "words": ["family", "kids", "children", "child", "cartoon", "animated"],
        "genres": ["children", "family", "kids", "anime"],
        "query": "family kids friendship adventure animated magical funny children",
    },
}

EMOJI_TO_MOOD = {e: m for m, d in MOODS.items() for e in d["emojis"]}
WORD_TO_MOOD = {w: m for m, d in MOODS.items() for w in d["words"]}


def resolve_mood(text):
    """Return the mood key for an emoji or typed phrase, or None."""
    text = (text or "").strip()
    for e, m in EMOJI_TO_MOOD.items():
        if e in text:
            return m
    low = text.lower()
    if low in MOODS:
        return low
    for token in low.replace(",", " ").split():
        if token in WORD_TO_MOOD:
            return WORD_TO_MOOD[token]
    for w, m in WORD_TO_MOOD.items():
        if w in low:
            return m
    return None


def recommend_by_mood(user_input, n=5, type_filter=None, bundle=None,
                      pool=25, shuffle=True, seed=None):
    """Return (mood_name_or_None, DataFrame of n suggestions)."""
    b = bundle or load_bundle()
    meta, X, vec = b["meta"], b["matrix"], b["vectorizer"]
    mood = resolve_mood(user_input)

    # Words the user typed (emoji stripped) are added to the query
    typed = "".join(ch for ch in (user_input or "") if ch.isascii()).strip()
    query = (MOODS[mood]["query"] + " " if mood else "") + typed
    qvec = vec.transform([query])
    sims = linear_kernel(qvec, X).ravel()

    allowed = np.ones(len(meta), dtype=bool)
    if mood:
        g = meta["listed_in"].str.lower()
        allowed = g.apply(lambda s: any(k in s for k in MOODS[mood]["genres"])).values
    if type_filter:
        allowed &= (meta["type"] == type_filter).values

    sims = np.where(allowed, sims, -1)
    top = np.argsort(sims)[::-1][:pool]
    top = [i for i in top if sims[i] > 0]
    if not top:
        return mood, pd.DataFrame()

    if shuffle:
        rng = random.Random(seed)
        top = rng.sample(top, k=min(n, len(top)))
    else:
        top = top[:n]

    out = meta.iloc[top][["title", "type", "release_year", "listed_in", "description"]].copy()
    out["match"] = sims[top].round(3)
    return mood, out.reset_index(drop=True)


if __name__ == "__main__":
    bundle = load_bundle()
    tests = ["😂", "😢", "😱", "😍", "🔥", "I feel tired and want something cozy",
             "curious", "😡", "xyz"]
    for t in tests:
        mood, recs = recommend_by_mood(t, n=4, bundle=bundle, seed=7)
        print(f"\n=== Input: {t!r}  ->  mood: {mood} ===")
        if recs.empty:
            print("   (no matches)")
        else:
            print(recs[["title", "type", "release_year", "listed_in"]].to_string(index=False))
