import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from sentence_transformers import SentenceTransformer

from src import config

# Load environment variables from the .env file
load_dotenv()


def initialise_llm() -> Groq:
    """Initialises the Groq LLM with core parameters from config."""

    api_key: str | None = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure it's set in your .env file."
        )

    return Groq(
        api_key=api_key,
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        additional_kwargs={
            "top_p": config.LLM_TOP_P,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
            "presence_penalty": config.LLM_PRESENCE_PENALTY,
        },
    )


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embedding_model() -> SentenceTransformer:
    """Loads the sentence-transformers model (cached inside the project)."""
    config.EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(
        config.EMBEDDING_MODEL_NAME,
        device=_get_device(),
        cache_folder=str(config.EMBEDDING_CACHE_PATH),
    )


def load_recipes() -> pd.DataFrame:
    """Loads the cleaned recipes table (must contain 'search_text' and 'doc')."""
    recipes = pd.read_parquet(config.RECIPES_PATH)
    missing = {"search_text", "doc", "minutes", "calories"} - set(recipes.columns)
    if missing:
        raise ValueError(f"Recipes file is missing columns: {missing}")
    return recipes


def load_embeddings(recipes: pd.DataFrame, embed_model: SentenceTransformer) -> np.ndarray:
    """
    Loads saved embeddings if they exist and match the recipes table,
    otherwise computes them from 'search_text' and saves them.
    Row i of the result is the vector of recipes.iloc[i].
    """
    path = config.EMBEDDINGS_PATH

    if path.exists():
        embeddings = np.load(path)
        if embeddings.shape[0] == len(recipes):
            return embeddings
        print("Saved embeddings don't match the recipes table, recomputing...")

    embeddings = embed_model.encode(
        recipes["search_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings

