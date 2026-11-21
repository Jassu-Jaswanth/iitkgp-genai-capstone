"""RAG pipeline on LangChain components: chunking strategies, embeddings, indexes, retrievers.

Corpus: data/processed/clauses.jsonl (from rag/curate.py), grouped back into sections.
Every chunk carries metadata {chunk_id, section_id, clause_ids, citation, url} so retrieval can be scored
at clause and section level regardless of how the text was split.

LangChain pieces: RecursiveCharacterTextSplitter (char and HF-tokenizer modes), OpenAIEmbeddings (upGrad
bge-small endpoint), HuggingFaceEmbeddings (local), FAISS, BM25Retriever, EnsembleRetriever (weighted RRF),
MultiVectorRetriever (parent-child), ContextualCompressionRetriever + CrossEncoderReranker.
"""

import json
import ssl
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import httpx
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever, MultiVectorRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.storage import InMemoryStore
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"
BGE_TOKENIZER = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------- corpus

@dataclass
class Section:
    section_id: str
    citation: str
    heading: str
    url: str
    source_type: str
    doc_title: str
    clauses: list[dict]

    def header(self) -> str:
        return f"[{self.citation}] {self.heading}"

    def text_with_spans(self) -> tuple[str, list[tuple[int, int, str]]]:
        """Section body text plus (start, end, clause_id) spans, for mapping chunks back to clauses."""
        parts, spans, pos = [], [], 0
        for c in self.clauses:
            t = clause_text(c)
            parts.append(t)
            spans.append((pos, pos + len(t), c["clause_id"]))
            pos += len(t) + 2
        return "\n\n".join(parts), spans


def clause_text(c: dict, with_lead: bool = True) -> str:
    marker = f"{c['marker']} " if c.get("marker") and c["marker"].startswith("(") else ""
    lead = f"{c['lead_in']} " if with_lead and c.get("lead_in") else ""
    return f"{marker}{lead}{c['text']}".strip()


@lru_cache
def load_sections(path: Path = PROCESSED / "clauses.jsonl") -> list[Section]:
    by: dict[str, Section] = {}
    for line in path.open():
        c = json.loads(line)
        s = by.get(c["section_id"])
        if s is None:
            s = by[c["section_id"]] = Section(c["section_id"], c["section_citation"], c["heading"], c["url"],
                                              c["source_type"], c["doc_title"], [])
        s.clauses.append(c)
    return list(by.values())


# ---------------------------------------------------------------- tokenizer-aware length

@lru_cache
def bge_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BGE_TOKENIZER)


def n_tokens(text: str) -> int:
    return len(bge_tokenizer().encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- chunking strategies

def _meta(sec: Section, chunk_id: str, clause_ids: list[str]) -> dict:
    return {"chunk_id": chunk_id, "section_id": sec.section_id, "clause_ids": clause_ids, "citation": sec.citation,
            "heading": sec.heading, "url": sec.url, "source_type": sec.source_type}


def split_sections(splitter, sections: list[Section], header: bool) -> list[Document]:
    """Split each section's text with a LangChain splitter; map chunk spans back to clause ids."""
    docs = []
    for sec in sections:
        body, spans = sec.text_with_spans()
        for i, d in enumerate(splitter.create_documents([body])):
            start = d.metadata.get("start_index", body.find(d.page_content))
            end = start + len(d.page_content)
            ids = [cid for s, e, cid in spans if s < end and e > start]
            text = f"{sec.header()}\n{d.page_content}" if header else d.page_content
            docs.append(Document(page_content=text, metadata=_meta(sec, f"{sec.section_id}#{i}", ids)))
    return docs


def clause_pack(sections: list[Section], max_tokens: int = 350, header: bool = True) -> list[Document]:
    """Structure-aware: pack consecutive whole clauses of ONE section up to max_tokens (bge tokens).
    Oversize clauses are split with a token-aware recursive splitter. Never crosses a section boundary."""
    big = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(bge_tokenizer(), chunk_size=max_tokens, chunk_overlap=40)
    docs = []
    for sec in sections:
        hdr = sec.header() if header else ""
        budget = max_tokens - (n_tokens(hdr) + 1 if header else 0)
        buf, ids, used, n = [], [], 0, 0

        def emit():
            nonlocal buf, ids, used, n
            if buf:
                body = "\n".join(buf)
                docs.append(Document(page_content=f"{hdr}\n{body}" if header else body,
                                     metadata=_meta(sec, f"{sec.section_id}#{n}", ids)))
                n += 1
            buf, ids, used = [], [], 0

        for c in sec.clauses:
            in_chunk = "\n".join(buf)
            lead = c.get("lead_in") or ""
            t = clause_text(c, with_lead=bool(lead) and lead[-120:] not in in_chunk)
            tok = n_tokens(t)
            if tok > budget:
                emit()
                for piece in big.split_text(t):
                    buf, ids, used = [piece], [c["clause_id"]], 0
                    emit()
                continue
            if used + tok > budget:
                emit()
                t = clause_text(c)  # new chunk: restore the lead-in so the clause stands alone
                tok = n_tokens(t)
            buf.append(t)
            ids.append(c["clause_id"])
            used += tok + 1
        emit()
    return docs


def clause_children(sections: list[Section]) -> tuple[list[Document], dict[str, Document]]:
    """Parent-child: one child per clause (with header) -> parent = whole section (truncated to ~700 tokens)."""
    trunc = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(bge_tokenizer(), chunk_size=700, chunk_overlap=0)
    children, parents = [], {}
    for sec in sections:
        body, _ = sec.text_with_spans()
        first = trunc.split_text(body)[0] if body else ""
        parents[sec.section_id] = Document(page_content=f"{sec.header()}\n{first}",
                                           metadata=_meta(sec, f"{sec.section_id}#parent", [c["clause_id"] for c in sec.clauses]))
        for c in sec.clauses:
            for j, piece in enumerate(trunc_child().split_text(clause_text(c))):  # oversize clauses -> pieces <= 450 tokens
                children.append(Document(page_content=f"{sec.header()}\n{piece}",
                                         metadata={**_meta(sec, f"{c['clause_id']}~{j}", [c["clause_id"]]),
                                                   "parent_id": sec.section_id}))
    return children, parents


@lru_cache
def trunc_child():
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(bge_tokenizer(), chunk_size=420, chunk_overlap=40)


def chunk_strategies() -> dict[str, Callable[[list[Section]], list[Document]]]:
    chars = lambda size, ov: RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=ov, add_start_index=True)
    toks = lambda size, ov: RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        bge_tokenizer(), chunk_size=size, chunk_overlap=ov, add_start_index=True)
    return {
        "recursive-450/80": lambda s: split_sections(chars(450, 80), s, header=False),    # course hands-on
        "recursive-600/120": lambda s: split_sections(chars(600, 120), s, header=False),  # course notebook default
        "token-256/32": lambda s: split_sections(toks(256, 32), s, header=False),         # tokenizer-aware
        "token-256/32+header": lambda s: split_sections(toks(256, 32), s, header=True),   # contextual header ablation
        "clause-pack-350": lambda s: clause_pack(s, 350, header=True),                    # ours, structure-aware
        "clause-pack-350-noheader": lambda s: clause_pack(s, 350, header=False),          # header ablation
    }


# ---------------------------------------------------------------- embeddings

def _env() -> dict[str, str]:
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


class QueryInstructionEmbeddings(Embeddings):
    """BGE v1.5 recommends an instruction prefix on short retrieval QUERIES (not passages)."""

    def __init__(self, base: Embeddings, instruction: str = BGE_QUERY_INSTRUCTION):
        self.base, self.instruction = base, instruction

    def embed_documents(self, texts):
        return self.base.embed_documents(texts)

    def embed_query(self, text):
        return self.base.embed_query(self.instruction + text)


def upgrad_embeddings() -> OpenAIEmbeddings:
    env = _env()
    ctx = ssl.create_default_context(cafile=str((ROOT / env["EMBEDDING_CA_CERT_PATH"]).resolve()))
    return OpenAIEmbeddings(model=env["EMBEDDING_MODEL"], base_url=env["EMBEDDING_BASE_URL"], api_key=env["EMBEDDING_API_KEY"],
                            check_embedding_ctx_length=False, chunk_size=16, max_retries=5, request_timeout=60,
                            http_client=httpx.Client(verify=ctx), http_async_client=httpx.AsyncClient(verify=ctx))


def local_embeddings(model: str = "BAAI/bge-small-en-v1.5"):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})


# ---------------------------------------------------------------- indexes and retrievers

def build_faiss(docs: list[Document], embeddings: Embeddings, name: str | None = None) -> FAISS:
    """FAISS flat inner-product index (vectors are unit-norm -> cosine). Cached on disk by name."""
    path = INDEX_DIR / name if name else None
    if path and (path / "index.faiss").exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True,
                                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    vs = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    if path:
        path.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(path))
    return vs


def retrievers_for(docs: list[Document], vs: FAISS, k: int, *, fetch_k: int = 30,
                   reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> dict[str, BaseRetriever]:
    dense = vs.as_retriever(search_kwargs={"k": k})
    dense_wide = vs.as_retriever(search_kwargs={"k": fetch_k})
    bm25 = BM25Retriever.from_documents(docs, k=k)
    bm25_wide = BM25Retriever.from_documents(docs, k=fetch_k)
    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5], c=60)
    hybrid_wide = EnsembleRetriever(retrievers=[bm25_wide, dense_wide], weights=[0.5, 0.5], c=60)
    rerank = ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder(reranker_model), top_n=k), base_retriever=hybrid_wide)
    return {"dense": dense, "bm25": bm25, "hybrid-rrf": hybrid, "hybrid+rerank": rerank}


@lru_cache
def cross_encoder(name: str) -> HuggingFaceCrossEncoder:
    return HuggingFaceCrossEncoder(model_name=name)


def parent_child_retriever(children: list[Document], parents: dict[str, Document], embeddings: Embeddings,
                           k: int, name: str | None = None) -> MultiVectorRetriever:
    vs = build_faiss(children, embeddings, name)
    store = InMemoryStore()
    store.mset(list(parents.items()))
    return MultiVectorRetriever(vectorstore=vs, docstore=store, id_key="parent_id", search_kwargs={"k": k * 4})


def dedupe_by_chunk(docs: list[Document], k: int) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        cid = d.metadata.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            out.append(d)
        if len(out) == k:
            break
    return out


def section_index(sections: list[Section]) -> dict[str, Section]:
    return {s.section_id: s for s in sections}


def clause_to_section(sections: list[Section]) -> dict[str, str]:
    m = {}
    for s in sections:
        for c in s.clauses:
            m[c["clause_id"]] = s.section_id
    return m


def stats(docs: list[Document]) -> dict:
    lens = sorted(n_tokens(d.page_content) for d in docs)
    by_sec = defaultdict(int)
    for d in docs:
        by_sec[d.metadata["section_id"]] += 1
    return {"chunks": len(docs), "median_tokens": lens[len(lens) // 2], "p95_tokens": lens[int(len(lens) * .95)],
            "max_tokens": lens[-1], "over_512": sum(1 for x in lens if x > 512), "sections": len(by_sec)}
