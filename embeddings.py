import os

from chromadb import Documents, EmbeddingFunction, Embeddings
from google.genai import types

from google import genai
from google.api_core import retry
from google.generativeai import configure, get_model
from dotenv import load_dotenv

# Configure Gemini API key
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

client = genai.Client(api_key=GOOGLE_API_KEY)

configure(api_key=GOOGLE_API_KEY)

# Retry logic
is_retriable = lambda e: hasattr(e, "code") and e.code in {429, 503}

class GeminiEmbeddingFunction(EmbeddingFunction):
    # def __init__(self, document_mode=True):
    #     self.document_mode = document_mode
    #     self.model = get_model("models/text-embedding-004")

    # @retry.Retry(predicate=is_retriable)
    # def __call__(self, input):
    #     if not isinstance(input, list):
    #         input = [input]
    #     task_type = "retrieval_document" if self.document_mode else "retrieval_query"
    #     response = self.model.embed_content(content=input, task_type=task_type)
    #     return [e.values for e in response.embeddings]
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]