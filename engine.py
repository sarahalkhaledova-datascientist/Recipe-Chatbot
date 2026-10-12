import re
from typing import List

import numpy as np
import pandas as pd
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src import config
from src.model_loader import (
    initialise_llm,
    load_embedding_model,
    load_embeddings,
    load_recipes,
)

CONTEXT_PROMPT = (
    "Here are the recipes retrieved for the user's latest question:\n\n"
    "{context_str}\n\n"
    "Use ONLY these recipes to answer. If the list above is empty or nothing "
    "fits the question, say you couldn't find a matching recipe and suggest "
    "rephrasing."
)


def parse_constraints(text: str):
    """Extracts simple limits like 'under 20 minutes' or 'under 300 calories'."""
    text = text.lower()
    max_minutes = max_calories = None

    m = re.search(
        r"(?:under|less than|within|below)\s+(\d+)\s*(hours?|hrs?|minutes?|mins?)", text
    )
    if m:
        n = int(m.group(1))
        max_minutes = n * 60 if m.group(2).startswith("h") else n

    c = re.search(r"(?:under|less than|below)\s+(\d+)\s*cal", text)
    if c:
        max_calories = int(c.group(1))

    return max_minutes, max_calories


class RecipeRetriever(BaseRetriever):
    """
    One recipe = one vector (serialization, no chunking).
    Search happens on the embeddings of 'search_text', but the text sent to
    the LLM is the full 'doc' (with the steps).

    Stage 1 of retrieval: filters (time / calories) + minimum cosine score.
    'top_k' is how many recipes to return (use a bigger number when a
    reranker will cut the list down afterwards).
    """

    def __init__(self, recipes, embeddings, embed_model, top_k: int = config.TOP_K,
             fallback_node: bool = False):
        
        self._recipes = recipes
        self._embeddings = embeddings
        self._embed_model = embed_model
        self._top_k = top_k
        super().__init__()
        self._fallback_node = fallback_node

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        max_minutes, max_calories = parse_constraints(query)

        q = self._embed_model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q.T).ravel()

        mask = np.ones(len(self._recipes), dtype=bool)
        if max_minutes:
            mask &= self._recipes["minutes"].values <= max_minutes
        if max_calories:
            mask &= self._recipes["calories"].values <= max_calories
        scores = np.where(mask, scores, -1)

        top = [
            i for i in np.argsort(-scores)[: self._top_k]
            if scores[i] >= config.MIN_SCORE
        ]

        # بـ _retrieve قبل return الأخير
        if not top and self._fallback_node:
       
          return [NodeWithScore(
             node=TextNode(text="(no matching recipes were found for this message)"),
             score=0.0,
          )]

        return [
            NodeWithScore(
                node=TextNode(text=self._recipes.iloc[i]["doc"]),
                score=float(scores[i]),
            )
            for i in top
        ]


def build_reranker(top_n: int | None = None) -> SentenceTransformerRerank:
    """Cross-encoder that re-orders the retrieved recipes and keeps the best top_n."""
    return SentenceTransformerRerank(
        model=config.RERANKER_MODEL_NAME,
        top_n=top_n or config.RERANKER_TOP_N,
    )


def build_chat_engine(
    use_reranker: bool | None = None,
    retriever_k: int | None = None,
    reranker_top_n: int | None = None,
) -> CondensePlusContextChatEngine:
    """
    Builds the full RAG chat engine (retriever + Groq LLM + summary memory).
    With use_reranker=True: fetch 'retriever_k' recipes, then keep the best
    'reranker_top_n'. Everything defaults to the values in src/config.py.
    """
    use_reranker = config.USE_RERANKER if use_reranker is None else use_reranker
    if retriever_k is None:
        retriever_k = config.RETRIEVER_K if use_reranker else config.TOP_K

    llm = initialise_llm()
    embed_model = load_embedding_model()
    recipes = load_recipes()
    embeddings = load_embeddings(recipes, embed_model)

    retriever = RecipeRetriever(recipes, embeddings, embed_model, top_k=retriever_k, fallback_node=True)

    postprocessors = []
    if use_reranker:
        postprocessors.append(build_reranker(reranker_top_n))

    # Keeps recent messages up to the token limit and summarizes older ones with the LLM.
    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=[],
        llm=llm,
        token_limit=config.CHAT_MEMORY_TOKEN_LIMIT,
    )

    # Rewrites follow-up questions ("how do I make the first one?") into
    # standalone questions before searching.
    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        memory=memory,
        system_prompt=config.LLM_SYSTEM_PROMPT,
        context_prompt=CONTEXT_PROMPT,
        node_postprocessors=postprocessors,
    )


def main_chat_loop():
    """Main chat loop for a conversational chatbot."""
    build_chat_engine().chat_repl()
