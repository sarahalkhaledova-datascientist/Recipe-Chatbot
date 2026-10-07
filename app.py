import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------- الإعدادات ----------
DATA_DIR = Path(__file__).parent / "Data"
LLM_MODEL = "openai/gpt-oss-20b"
MIN_SCORE = 0.35
TOP_K = 3
MAX_HISTORY_MESSAGES = 6   # آخر 3 أسئلة وأجوبة فقط، لتوفير الـ tokens

SYSTEM_PROMPT = """You are a friendly recipe assistant.
Answer ONLY using the recipes provided in the context.
- Recommend 1 to 3 recipes that fit the question. For each one, use the exact recipe name and give one short sentence with its time and calories and why it fits.
- If the user asks how to cook one, give its ingredients and steps exactly as written in the context.
- The ingredient lists have NO quantities. Only mention an amount if it appears in the steps text. Otherwise say "amount not specified".
- Never invent recipes, ingredients, steps or numbers that are not in the context.
- If the question is not about food or cooking, say you can only help with recipes."""

st.set_page_config(page_title="Recipe Chatbot", page_icon="🍳")


# ---------- تحميل البيانات والنموذج (مرة وحدة فقط) ----------

st.write("DATA_DIR:", str(DATA_DIR), "exists:", DATA_DIR.exists())
st.write("Repo files:", sorted(p.name for p in DATA_DIR.parent.iterdir()))

@st.cache_resource(show_spinner="Loading recipes and embedding model...")
def load_resources():
    sample = pd.read_parquet(DATA_DIR / "recipes_slim.parquet")
    embeddings = np.load(DATA_DIR / "embeddings_f16.npy")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return sample, embeddings, model


sample, embeddings, embed_model = load_resources()


# ---------- البحث ----------
def parse_constraints(text):
    text = text.lower()
    max_minutes = max_calories = None

    m = re.search(r"(?:under|less than|within|below)\s+(\d+)\s*(hours?|hrs?|minutes?|mins?)", text)
    if m:
        n = int(m.group(1))
        max_minutes = n * 60 if m.group(2).startswith("h") else n

    c = re.search(r"(?:under|less than|below)\s+(\d+)\s*cal", text)
    if c:
        max_calories = int(c.group(1))

    return max_minutes, max_calories


def search(query, k=TOP_K):
    max_minutes, max_calories = parse_constraints(query)
    q = embed_model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q.T).ravel()

    mask = np.ones(len(sample), dtype=bool)
    if max_minutes:
        mask &= sample["minutes"].values <= max_minutes
    if max_calories:
        mask &= sample["calories"].values <= max_calories
    scores = np.where(mask, scores, -1)

    top = [i for i in np.argsort(-scores)[:k] if scores[i] >= MIN_SCORE]
    return sample.iloc[top].assign(score=scores[top])


# ---------- حالة الجلسة ----------
if "messages" not in st.session_state:
    st.session_state.messages = []   # للعرض على الشاشة
if "history" not in st.session_state:
    st.session_state.history = []    # للإرسال إلى النموذج
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []  # آخر وصفات انجابت (لأسئلة المتابعة)


def reset_chat():
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.last_docs = []


# ---------- الشريط الجانبي ----------
with st.sidebar:
    st.header("Settings")
    user_key = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave empty to use the app's built-in key",
    )
    st.caption(f"Model: `{LLM_MODEL}`")
    st.caption(f"Recipes in index: {len(sample):,}")
    st.button("Clear chat", on_click=reset_chat)

    st.markdown("**Try asking:**")
    st.markdown(
        "- quick vegetarian pasta under 30 minutes\n"
        "- low calorie chicken salad under 300 calories\n"
        "- what can I cook with eggs, spinach and cheese?\n"
        "- how do I make the first one?"
    )

def get_server_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

api_key = user_key or get_server_key()

# ---------- الواجهة ----------
st.title("🍳 Recipe Chatbot")
st.caption("Ask for recipes by ingredients, time, calories or type of dish.")


def show_sources(sources):
    if not sources:
        return
    with st.expander("Recipes retrieved"):
        for s in sources:
            st.write(f"**{s['name']}** — {s['minutes']} min, {s['calories']:.0f} cal (match {s['score']:.2f})")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))

question = st.chat_input("What do you want to cook?")

if question:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = search(question)

            # وصفات جديدة: نحدّث السياق. غير هيك: نكمّل على آخر وصفات (أسئلة المتابعة)
            if len(results) > 0:
                st.session_state.last_docs = results["doc"].tolist()

            context = (
                "\n\n---\n\n".join(st.session_state.last_docs)
                if st.session_state.last_docs
                else "No matching recipes found."
            )
            user_msg = f"Recipes:\n{context}\n\nQuestion: {question}"

            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + st.session_state.history[-MAX_HISTORY_MESSAGES:]
                + [{"role": "user", "content": user_msg}]
            )

            try:
                client = Groq(api_key=api_key)
                resp = client.chat.completions.create(model=LLM_MODEL, messages=messages)
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"Sorry, something went wrong calling Groq: `{e}`"

        st.markdown(answer)

        sources = [
            {"name": r["name"], "minutes": r["minutes"], "calories": r["calories"], "score": r["score"]}
            for _, r in results.iterrows()
        ]
        show_sources(sources)

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
