import io
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

from openai import OpenAI

# Initialize OpenAI client (expects OPENAI_API_KEY in environment or st.secrets)
client = OpenAI()

# ----------------------------
# Utility and Processing
# ----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def split_text(text: str, chunk_size_words: int = 400, overlap_words: int = 50) -> List[str]:
    text = clean_text(text)
    words = text.split()
    if not words:
        return []
    if chunk_size_words <= 0:
        chunk_size_words = 400
    if overlap_words < 0:
        overlap_words = 0
    if overlap_words >= chunk_size_words:
        overlap_words = max(0, chunk_size_words // 4)
    step = chunk_size_words - overlap_words if chunk_size_words > overlap_words else chunk_size_words

    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def read_txt(file_bytes: bytes, encoding: str = "utf-8") -> str:
    try:
        return file_bytes.decode(encoding, errors="ignore")
    except Exception:
        try:
            return file_bytes.decode("latin-1", errors="ignore")
        except Exception:
            return ""


def read_pdf(uploaded_file) -> str:
    if PdfReader is None:
        st.error("PDF support requires 'pypdf' package. Please install it: pip install pypdf")
        return ""
    try:
        reader = PdfReader(uploaded_file)
        pages_text = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages_text.append(txt)
        return "\n".join(pages_text)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


def read_docx(uploaded_file) -> str:
    if Document is None:
        st.error("DOCX support requires 'python-docx'. Install it: pip install python-docx")
        return ""
    try:
        file_bytes = uploaded_file.read()
        bio = io.BytesIO(file_bytes)
        doc = Document(bio)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    except Exception as e:
        st.error(f"Failed to read DOCX: {e}")
        return ""


def read_csv(uploaded_file) -> str:
    # Try pandas first
    if pd is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df.to_csv(index=False)
        except Exception:
            uploaded_file.seek(0)  # reset pointer
    # Fallback: raw decode
    try:
        file_bytes = uploaded_file.read()
        return read_txt(file_bytes)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return ""


def read_file(uploaded_file) -> Tuple[str, str]:
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    text = ""
    if ext in [".txt", ".md", ".markdown", ".log"]:
        text = read_txt(uploaded_file.read())
    elif ext == ".pdf":
        text = read_pdf(uploaded_file)
    elif ext in [".docx", ".doc"]:
        text = read_docx(uploaded_file)
    elif ext in [".csv", ".tsv"]:
        text = read_csv(uploaded_file)
    else:
        # Fallback: try to decode as text
        try:
            text = read_txt(uploaded_file.read())
        except Exception:
            text = ""
    return clean_text(text), ext


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    # The API supports batching a list of strings
    res = client.embeddings.create(model=model, input=texts)
    vectors = [d.embedding for d in res.data]
    return np.array(vectors, dtype=np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_index(text: str, chunk_words: int, overlap_words: int) -> Dict[str, Any]:
    chunks = split_text(text, chunk_words, overlap_words)
    if not chunks:
        return {"chunks": [], "embeddings": np.zeros((0, 1536), dtype=np.float32)}
    embeddings = embed_texts(chunks)
    embeddings_norm = normalize_rows(embeddings)
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "embeddings_norm": embeddings_norm,
    }


def retrieve(query: str, index: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    if not index or not index.get("chunks"):
        return []
    q_emb = embed_texts([query])[0:1]
    if q_emb.shape[0] == 0:
        return []
    q_emb_norm = normalize_rows(q_emb)[0]
    doc_norm = index["embeddings_norm"]
    sims = doc_norm @ q_emb_norm  # cosine similarity
    top_k = max(1, min(top_k, sims.shape[0]))
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for rank, i in enumerate(idxs, start=1):
        results.append({
            "rank": rank,
            "index": int(i),
            "score": float(sims[i]),
            "text": index["chunks"][i],
        })
    return results


def build_context_snippets(retrieved: List[Dict[str, Any]]) -> str:
    lines = []
    for r in retrieved:
        lines.append(f"[{r['rank']}] {r['text']}")
    return "\n\n".join(lines)


def generate_answer(query: str,
                    context_snippets: str,
                    model: str = "gpt-4",
                    temperature: float = 0.2,
                    max_tokens: int = 700) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the Context provided. "
        "If the answer cannot be found in the context, reply that you don't know. "
        "When relevant, cite sources by referencing the bracketed snippet numbers like [1], [2]."
    )
    user_prompt = (
        f"Context:\n{context_snippets}\n\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"- Use only the context to answer.\n"
        f"- If insufficient information, say you don't know.\n"
        f"- Include inline citations [n] tied to snippets when appropriate."
    )

    response = client.chat.completions.create(
        model=model,  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return response.choices[0].message.content


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="File-based RAG QA", page_icon="🔎", layout="wide")
st.title("🔎 File-based RAG: Ask Questions About Your Document")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Top-K Snippets", 1, 10, 5, 1)
    chunk_words = st.slider("Chunk Size (words)", 100, 1200, 400, 50)
    overlap_words = st.slider("Chunk Overlap (words)", 0, 400, 50, 10)
    st.markdown("---")
    st.caption("Provide your OpenAI API key as environment variable OPENAI_API_KEY.")

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "retrievals" not in st.session_state:
    st.session_state.retrievals = {}
if "history" not in st.session_state:
    st.session_state.history = []

# File uploader
uploaded_file = st.file_uploader("Upload a document (txt, md, pdf, docx, csv)", type=["txt", "md", "markdown", "pdf", "docx", "csv", "tsv", "log"])

col1, col2 = st.columns([1, 1])
with col1:
    process_clicked = st.button("Process File", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear Index", use_container_width=True)

if clear_clicked:
    st.session_state.index = None
    st.session_state.file_name = None
    st.session_state.retrievals = {}
    st.session_state.history = []
    st.success("Cleared current index and history.")

if process_clicked:
    if not uploaded_file:
        st.warning("Please upload a file first.")
    else:
        with st.spinner("Reading and indexing the document..."):
            text, ext = read_file(uploaded_file)
            if not text.strip():
                st.error("Could not extract text from the file.")
            else:
                idx = build_index(text, chunk_words=chunk_words, overlap_words=overlap_words)
                if not idx["chunks"]:
                    st.error("No content chunks were created. Try adjusting chunk size.")
                else:
                    st.session_state.index = idx
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.history = []
                    st.success(f"Indexed {len(idx['chunks'])} chunks from '{uploaded_file.name}'.")

if st.session_state.index:
    st.info(f"Indexed file: {st.session_state.file_name} | Chunks: {len(st.session_state.index['chunks'])}")
else:
    st.info("Upload and process a file to enable question answering.")

# Query input
st.subheader("Ask a question about your file")
query = st.text_input("Enter your question", placeholder="e.g., What are the key findings?")
ask_clicked = st.button("Ask", type="secondary")

if ask_clicked:
    if not st.session_state.index:
        st.warning("Please upload and process a file first.")
    elif not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Retrieving relevant snippets and generating answer..."):
            retrieved = retrieve(query, st.session_state.index, top_k=top_k)
            if not retrieved:
                st.warning("No relevant context found in the index.")
            else:
                ctx = build_context_snippets(retrieved)
                answer = generate_answer(query, ctx, model=model_choice, temperature=temperature, max_tokens=700)
                st.session_state.history.append({
                    "query": query,
                    "answer": answer,
                    "retrieved": retrieved,
                })

# Display QA history
if st.session_state.history:
    st.subheader("Results")
    for i, item in enumerate(reversed(st.session_state.history), start=1):
        st.write(f"Q{i}: {item['query']}")
        st.write("A:")
        st.write(item["answer"])
        with st.expander("Show retrieved snippets and scores"):
            for r in item["retrieved"]:
                st.write(f"[{r['rank']}] (score: {r['score']:.4f})")
                st.write(r["text"][:1000] + ("..." if len(r["text"]) > 1000 else ""))
            st.caption("Higher score indicates higher semantic similarity.")
else:
    st.caption("Ask a question after indexing to see results here.")