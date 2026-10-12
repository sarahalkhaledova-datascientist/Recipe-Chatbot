from pathlib import Path

# --- LLM Model Configuration ---
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_MAX_NEW_TOKENS: int = 2000
LLM_TEMPERATURE: float = 0.3
LLM_TOP_P: float = 0.95
LLM_FREQUENCY_PENALTY: float = 0
LLM_PRESENCE_PENALTY: float = 0
LLM_SYSTEM_PROMPT: str =  """You are a friendly recipe assistant.
Answer ONLY using the recipes provided in the context.
- Recommend 1 to 3 recipes that fit the question. For each one, use the exact recipe name and give one short sentence with its time and calories and why it fits.
- If the user asks how to cook one, give its ingredients and steps exactly as written in the context.
- The ingredient lists have NO quantities. Only mention an amount if it appears in the steps text. Otherwise say "amount not specified".
- Never invent recipes, ingredients, steps or numbers that are not in the context.
- If the question is not about food or cooking, say you can only help with recipes.
- Answer directly with the facts only. Do not add commentary about the question or the match.
- You only see a few recipes, never the whole collection. Never state totals or counts, and never claim one recipe is the "best" or "healthiest" overall; say you can only suggest from the matches you found.
- Do not add cooking advice, substitutions or health claims that are not in the recipe text; if asked, say the recipe doesn't cover it.
- Never confirm that a recipe is safe for an allergy or diet. Only say what the ingredient list shows and advise checking the labels.
- If the user greets you or thanks you, reply briefly and ask what they would like to cook.
- If the message is not about food, politely say you can only help with recipes."""



# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


# --- Retrieval Configuration ---
# Each recipe is ONE document (serialization, no chunking)
TOP_K: int = 3                 # how many recipes to retrieve
MIN_SCORE: float = 0.35        # minimum cosine similarity; below this = "no match"


# --- Chat Memory Configuration ---

CHAT_MEMORY_TOKEN_LIMIT: int = 3900

# --- Data files ---
RECIPES_FILE: str = "recipes_sample.parquet"
EMBEDDINGS_FILE: str = "embeddings.npy"


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data"
RECIPES_PATH: Path = DATA_PATH / RECIPES_FILE
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage" / "embedding_model"
EMBEDDINGS_PATH: Path = ROOT_PATH / "local_storage" / "embeddings" / EMBEDDINGS_FILE

# --- Reranker Configuration ---
USE_RERANKER: bool = False     # فعّليه بعد ما تثبتي إنه بيحسّن
RETRIEVER_K: int = 10          # كم وصفة بيجيب البحث قبل الـ rerank
RERANKER_TOP_N: int = 3        # كم وصفة بتضل للنموذج
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"