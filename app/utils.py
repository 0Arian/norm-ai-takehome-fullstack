from pydantic import BaseModel
import qdrant_client
from llama_index.core import Document as LIDocument
from llama_index.readers.file import PyMuPDFReader
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple
import fitz  # PyMuPDF
import pymupdf4llm
import os

# LlamaIndex (core + OpenAI + Qdrant vector store)
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate, get_response_synthesizer
from llama_index.core.query_engine import CitationQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

key = os.environ['OPENAI_API_KEY']

@dataclass
class Input:
    query: str
    file_path: str

@dataclass
class Citation:
    source: str
    text: str

class Output(BaseModel):
    query: str
    response: str
    citations: list[Citation]

class DocumentService:
    """
    Update this service to load the pdf and extract its contents.
    The example code below will help with the data structured required
    when using the QdrantService.load() method below. Note: for this
    exercise, ignore the subtle difference between llama-index's 
    Document and Node classes (i.e, treat them as interchangeable).

    # example code
    def create_documents() -> list[Document]:

        docs = [
            Document(
                metadata={"Section": "Law 1"},
                text="Theft is punishable by hanging",
            ),
            Document(
                metadata={"Section": "Law 2"},
                text="Tax evasion is punishable by banishment.",
            ),
        ]

        return docs

     """

    def __init__(self, default_pdf_path: Optional[str] = None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.pdf_path = (
            default_pdf_path or os.path.join(base_dir, "docs", "laws.pdf")
        )

    def create_documents(self) -> List[LIDocument]:
        """
        Simple parser:
        - Read markdown from PDF
        - Ignore trailing Citations
        - Treat bold lines like "**4.1.** **Trials of the Crown**" as heading markers
        - Start a new document at each numbered line "4.1.1. ..."
        - Append subsequent lines until the next numbered line
        - Title for each document is the current bold heading's title
        """
        md_text = pymupdf4llm.to_markdown(self.pdf_path) if self.pdf_path else ""
        if not md_text:
            return []

        # Remove trailing citations block if present
        m = re.search(r"(\*\*Citations:\*\*|Citations:)", md_text, flags=re.IGNORECASE)
        content_text = md_text[: m.start()] if m else md_text

        heading_re = re.compile(r"^\*\*(\d+(?:\.\d+)*)\.\*\*\s+\*\*(.*?)\*\*\s*$")
        number_re = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.*\S)?\s*$")

        current_title_prefix: Optional[str] = None
        current_title_text: Optional[str] = None

        current_section_number: Optional[str] = None
        current_section_title: Optional[str] = None
        current_lines: List[str] = []

        docs: List[LIDocument] = []

        def finalize_current() -> None:
            nonlocal current_section_number, current_lines, current_section_title
            if current_section_number is None:
                return
            body_text = "\n".join([ln.rstrip() for ln in current_lines]).strip()
            if not body_text:
                # Skip empty bodies
                current_section_number = None
                current_lines = []
                current_section_title = None
                return
            docs.append(
                LIDocument(
                    metadata={
                        "Section": f"Law {current_section_number}",
                        "Title": current_section_title or None,
                    },
                    text=body_text,
                )
            )
            current_section_number = None
            current_section_title = None
            current_lines = []

        for raw_line in content_text.splitlines():
            line = raw_line.rstrip("\n")

            # Bold heading line updates the current title context
            mh = heading_re.match(line.strip())
            if mh:
                # Finish any in-progress section before switching headings
                finalize_current()
                current_title_prefix = mh.group(1)
                current_title_text = mh.group(2).strip()
                continue

            # Numbered content line starts a new document
            mn = number_re.match(line)
            if mn:
                # Finish previous section
                finalize_current()
                sec_num = mn.group(1)
                first_line = (mn.group(2) or "").rstrip()

                # Assign title from nearest heading context
                title_for_section: Optional[str] = None
                if current_title_prefix and (
                    sec_num == current_title_prefix or sec_num.startswith(current_title_prefix + ".")
                ):
                    title_for_section = current_title_text

                current_section_number = sec_num
                current_section_title = title_for_section
                current_lines = [first_line] if first_line else []
                continue

            # Accumulate wrapped/continuation lines for the current section
            if current_section_number is not None:
                current_lines.append(line.rstrip())

        # Finalize last section if any
        finalize_current()

        # Fallback: if nothing was parsed but content exists, store a single doc
        if not docs and content_text.strip():
            docs.append(LIDocument(metadata={"Section": "Law 1"}, text=content_text.strip()))

        return docs

class QdrantService:
    def __init__(self, k: int = 2, system_prompt: Optional[str] = "Make sure to answer the question based on the context provided. If you don't know the answer, say so."):
        self.k = k
        self.system_prompt = system_prompt
        self.client = None
        self.storage_context: Optional[StorageContext] = None
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[CitationQueryEngine] = None

    def connect(self) -> None:
        # Initialize in-memory Qdrant and bind as LlamaIndex vector store
        self.client = qdrant_client.QdrantClient(location=":memory:")

        # Configure LLM and embeddings globally for LlamaIndex
        Settings.llm = LIOpenAI(api_key=key, model="gpt-5-nano", temperature=0)
        Settings.embed_model = OpenAIEmbedding(
            api_key=key,
            model="text-embedding-3-small",
        )

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="temp",
        )
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

    def load(self, docs: list[LIDocument]) -> None:
        if self.storage_context is None:
            raise RuntimeError("Vector store is not initialized. Call connect() first.")
        if not docs:
            return

        # Build or rebuild the LlamaIndex index on top of Qdrant
        self.index = VectorStoreIndex.from_documents(
            docs,
            storage_context=self.storage_context,
        )

        # Prepare a query engine; if a system prompt is provided, use a custom prompt
        if self.system_prompt:
            qa_prompt_tmpl = (
                f"{self.system_prompt}\n"
                "Context information is below.\n"
                "-------------------------------\n"
                "{context_str}\n"
                "-------------------------------\n"
                "Given the context information and not prior knowledge, answer the query.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_prompt = PromptTemplate(qa_prompt_tmpl)

            retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.k)
            response_synthesizer = get_response_synthesizer(text_qa_template=qa_prompt)
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )
        else:
            # Default: citation-enabled engine with top-k control
            self.query_engine = CitationQueryEngine.from_args(
                self.index,
                similarity_top_k=self.k,
            )

    def query(self, query_str: str) -> Output:
        if self.index is None:
            raise RuntimeError("Service not connected or loaded. Call connect() and load() first.")

        # Use an existing engine or make a fresh one honoring the current k
        engine = self.query_engine or CitationQueryEngine.from_args(
            self.index,
            similarity_top_k=self.k,
        )

        resp = engine.query(query_str)
        response_text = getattr(resp, "response", None) or str(resp)

        citations: list[Citation] = []
        # Limit citations to top-k retrieved source nodes to reflect retrieval depth
        for sn in (getattr(resp, "source_nodes", None) or [])[: self.k]:
            node = getattr(sn, "node", None) or sn
            metadata = getattr(node, "metadata", None) or {}
            source = metadata.get("Section") or (
                f"Page {metadata.get('page')}" if metadata.get("page") else "Unknown"
            )
            text = getattr(node, "text", "")
            citations.append(Citation(source=source, text=text))

        return Output(query=query_str, response=response_text, citations=citations)

