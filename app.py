import os

import streamlit as st

st.set_page_config(page_title="Recipe Chatbot", page_icon="🍳")

# --- API key: Streamlit secrets first, then environment / .env ---
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass   # fall back to the .env file (loaded by src.model_loader)

from src import config  # noqa: E402


@st.cache_resource(show_spinner="Loading recipes and models...")
def get_engine():
    """Builds the RAG chat engine once and reuses it between reruns."""
    from src.engine import build_chat_engine
    return build_chat_engine()


def recipe_sources(response) -> list[tuple[str, float | None]]:
    """(recipe name, score) for every retrieved recipe (skips the 'no match' placeholder)."""
    sources = []
    for node in response.source_nodes:
        text = node.node.get_content()
        if text.startswith("Recipe:"):
            name = text.split("\n", 1)[0].replace("Recipe:", "").strip()
            sources.append((name, node.score))
    return sources


try:
    engine = get_engine()
except Exception as e:
    st.error(f"Could not start the chatbot: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_chat():
    st.session_state.messages = []
    engine.reset()   # also clears the engine's memory


with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: `{config.LLM_MODEL}`")
    st.caption(f"Reranker: {'on' if config.USE_RERANKER else 'off'}")
    st.button("Clear chat", on_click=clear_chat)
    st.markdown("**Try asking:**")
    st.markdown(
        "- quick vegetarian pasta under 30 minutes\n"
        "- what can I cook with eggs, spinach and cheese?\n"
        "- how do I make the first one?"
    )

st.title("🍳 Recipe Chatbot")
st.caption("Ask for recipes by ingredients, time, calories or type of dish.")


def show_sources(sources):
    if not sources:
        return
    with st.expander("Recipes retrieved"):
        for name, score in sources:
            st.write(f"**{name}**" + (f" (score {score:.2f})" if score is not None else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))

question = st.chat_input("What do you want to cook?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = engine.chat(question)
                answer = str(response).strip() or "Sorry, I couldn't come up with an answer."
                sources = recipe_sources(response)
            except Exception as e:
                answer, sources = f"Sorry, something went wrong: `{e}`", []
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
