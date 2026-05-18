from __future__ import annotations
import uuid
from typing import Dict, List, Optional
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Document:
    def __init__(self, text: str, metadata: Dict[str, str]):
        self.text = text
        self.metadata = metadata

class SearchResult:
    def __init__(self, score: float, document: Document):
        self.score = score
        self.document = document

class FilteredVectorStore:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []

    def add_documents(self, documents: List[Document]) -> None:
        texts = [doc.text for doc in documents]
        new_embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)

    def search(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, str]] = None,) -> List[SearchResult]:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

        filtered_docs: List[Document] = []
        filtered_embeddings: List[np.ndarray] = []

        for doc, emb in zip(self.documents, self.embeddings):
            include_doc = True

            if metadata_filter is not None:
                for key, value in metadata_filter.items():
                    doc_value = doc.metadata.get(key)
                    
                    # Support range for years
                    if isinstance(value, (tuple, list)) and len(value) == 2:
                        try:
                            doc_num = float(doc_value)
                            min_val, max_val = float(value[0]), float(value[1])
                            if not (min_val <= doc_num <= max_val):
                                include_doc = False
                                break
                        except (ValueError, TypeError):
                            include_doc = False
                            break
                    # contains string and lower for artist and song fields
                    elif key in ['artist', 'song']:
                        if not (isinstance(value, str) and isinstance(doc_value, str) and 
                                value.lower() in doc_value.lower()):
                            include_doc = False
                            break
                    # Exact match for other fields
                    elif doc_value != value:
                        include_doc = False
                        break

            if include_doc:
                filtered_docs.append(doc)
                filtered_embeddings.append(emb)



        if not filtered_docs:
            return []

        scores = cosine_similarity(query_embedding, np.array(filtered_embeddings))[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [SearchResult(scores[i], filtered_docs[i]) for i in top_indices]

class Metadata(BaseModel):
    # schema for song lyrics
    song: str
    artist: str
    year: str
    rank: str
    source: str


class LyricsRequest(BaseModel):
    text: str
    metadata: Metadata


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    metadata_filter: Optional[Dict[str, str]] = None


CHUNK_SIZE = 400
CHUNK_THRESHOLD = 500

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

app = FastAPI(
    title="Semantic Song Lyrics Search API",
    description="Ingest song lyrics, chunk them, and query by semantic similarity.",
    version="1.0.0",
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

vector_store = FilteredVectorStore(embedding_model)
lyrics_db: Dict[str, dict] = {}  # document_id -> {text, metadata}

@app.post("/lyrics", status_code=201)
def create_lyrics(request: LyricsRequest):
    document_id = str(uuid.uuid4())

    # Persist the nonchunked document
    lyrics_db[document_id] = {
        "text": request.text,
        "metadata": request.metadata.model_dump(),
    }

    if len(request.text) > CHUNK_THRESHOLD:
        chunks = split_into_chunks(request.text)
    else:
        chunks = [request.text]

    # vector the chunks
    vector_docs: List[Document] = []
    base_metadata = request.metadata.model_dump()

    for chunk in chunks:
        # help from copiot
        # ** unpacks dict; merge base_metadata with new document_id key
        chunk_metadata = {**base_metadata, "document_id": document_id}
        vector_docs.append(Document(text=chunk, metadata=chunk_metadata))

    vector_store.add_documents(vector_docs)

    return {
        "document_id": document_id,
        "chunks_indexed": len(chunks),
        "message": "Lyrics ingested successfully.",
    }


@app.get("/lyrics/{document_id}")
def get_lyrics(document_id: str):
    if document_id not in lyrics_db:
        raise HTTPException(status_code=404, detail="Lyrics not found.")

    doc = lyrics_db[document_id]
    return {
        "document_id": document_id,
        "text": doc["text"],
        "metadata": doc["metadata"],
    }


@app.post("/lyrics/search")
def search_lyrics(request: SearchRequest):
    # Returns each matching chunk with score and metadata
    results = vector_store.search(
        query=request.query,
        top_k=request.top_k,
        metadata_filter=request.metadata_filter,
    )

    return {
        "query": request.query,
        "results": [
            {
                "similarity_pct": float(round(r.score * 100, 2)),
                "chunk": r.document.text,
                "metadata": r.document.metadata,
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)