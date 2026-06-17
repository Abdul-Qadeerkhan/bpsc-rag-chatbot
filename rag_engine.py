"""
RAG Engine — Core Logic
Uses Groq (free, fast) + ChromaDB (local vector store) + LangChain
"""

import os
import tempfile
from typing import Tuple, List

# Vector store & embeddings
import chromadb
from chromadb.utils import embedding_functions

# LLM
from groq import Groq

# Document processing
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ── System prompt tuned for BPSC / Pakistan context ──────────────
SYSTEM_PROMPT = """You are an expert AI assistant specializing in:
- BPSC (Balochistan Public Service Commission) exam preparation
- Pakistan Studies, History, Geography, General Knowledge
- Balochistan regional knowledge (culture, districts, governance)
- General Science and Current Affairs relevant to Pakistan

Your role:
1. Answer questions clearly and accurately based on provided context
2. If context is available, cite it and build your answer from it
3. If no context is given, use your training knowledge about Pakistan/BPSC topics
4. Be concise but complete — students need precise answers for exams
5. When relevant, mention which paper/topic this falls under (e.g., Pakistan Studies, General Knowledge)
6. Support both English and Urdu questions

Format your answers clearly with:
- Direct answer first
- Key points or sub-points if needed  
- Exam tip when relevant (mark it as 💡 Exam Tip:)
"""


class RAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.groq_client = Groq(api_key=api_key)
        self.collection = None
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB in-memory (no persistence needed for hackathon)"""
        self.chroma_client = chromadb.Client()
        try:
            self.chroma_client.delete_collection("bpsc_docs")
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(
            name="bpsc_docs",
            embedding_function=self.ef
        )

    def _extract_text(self, file) -> str:
        """Extract text from PDF or TXT file"""
        filename = file.name.lower()
        if filename.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            reader = PdfReader(tmp_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            os.unlink(tmp_path)
            return text
        elif filename.endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")
        return ""

    def ingest_files(self, files) -> dict:
        """Process uploaded files, chunk, and embed into ChromaDB"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "۔", ".", " "]  # includes Urdu full stop
        )

        all_chunks = []
        all_ids = []
        all_metadatas = []
        doc_count = 0

        for file in files:
            text = self._extract_text(file)
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{file.name}_{i}")
                all_metadatas.append({"source": file.name, "chunk_index": i})
            doc_count += 1

        if all_chunks:
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                self.collection.add(
                    documents=all_chunks[i:i+batch_size],
                    ids=all_ids[i:i+batch_size],
                    metadatas=all_metadatas[i:i+batch_size]
                )

        return {"doc_count": doc_count, "chunk_count": len(all_chunks)}

    def query(self, question: str, top_k: int = 4) -> Tuple[str, List[str]]:
        """Retrieve relevant chunks and generate answer"""
        # Step 1: Retrieve
        results = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, self.collection.count())
        )

        context_chunks = results["documents"][0] if results["documents"] else []
        sources = []
        if results["metadatas"] and results["metadatas"][0]:
            sources = list({m["source"] for m in results["metadatas"][0]})

        # Step 2: Generate
        context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No documents uploaded."

        prompt = f"""Context from study material:
{context_text}

Student Question: {question}

Answer based on the context above. If context doesn't cover this, use your knowledge about Pakistan/BPSC topics."""

        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
        )

        answer = response.choices[0].message.content
        return answer, sources

    def answer_without_docs(self, question: str) -> Tuple[str, List[str]]:
        """Answer using only LLM knowledge (no documents uploaded)"""
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=800,
        )
        answer = response.choices[0].message.content
        return answer, []
