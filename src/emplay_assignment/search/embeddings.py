
from __future__ import annotations

import logging

from google import genai
from google.genai import types

from emplay_assignment.config import settings


logger = logging.getLogger(__name__)


def create_gemini_client() -> genai.Client:
    """Create an authenticated Gemini API client."""
    api_key = settings.gemini_api_key.get_secret_value().strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your local .env file."
        )

    return genai.Client(api_key=api_key)


def prepare_document_text(
    text: str,
    title: str | None = None,
) -> str:
    """Prepare a document using Gemini Embedding 2's retrieval format."""
    clean_title = (title or "none").strip()
    clean_text = text.strip()

    return f"title: {clean_title} | text: {clean_text}"


def prepare_query_text(query: str) -> str:
    """Prepare a question for retrieval-based question answering."""
    clean_query = query.strip()

    if not clean_query:
        raise ValueError("Query cannot be empty.")

    return f"task: question answering | query: {clean_query}"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple independent texts in one Gemini API request."""
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Embedding inputs cannot contain empty text.")

    client = create_gemini_client()

    
    contents = [
        types.Content(
            parts=[types.Part.from_text(text=text)]
        )
        for text in texts
    ]

    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimension
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    if len(response.embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, "
            f"received {len(response.embeddings)}."
        )

    vectors: list[list[float]] = []

    for index, embedding in enumerate(response.embeddings):
        if embedding.values is None:
            raise RuntimeError(
                f"Gemini returned an empty vector at index {index}."
            )

        vector = [float(value) for value in embedding.values]

        if len(vector) != settings.embedding_dimension:
            raise RuntimeError(
                f"Dimension mismatch at index {index}: "
                f"expected {settings.embedding_dimension}, "
                f"received {len(vector)}."
            )

        vectors.append(vector)

    logger.info("Generated %s embeddings.", len(vectors))
    return vectors
