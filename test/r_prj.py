"""RAG pipeline: index, retrieve, generate, and evaluate over a codebase."""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fire
import numpy as np
import torch
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MinimalSource(BaseModel):
    """A location in the corpus defined by file path and character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without ground-truth sources or answer."""

    question_id: str
    question: str


class AnsweredQuestion(BaseModel):
    """A question with ground-truth sources and a reference answer."""

    question_id: str
    question: str
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(BaseModel):
    """Retrieved sources plus the generated answer for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]
    answer: str


class StudentSearchResults(BaseModel):
    """Search output for a whole dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search and answer output for a whole dataset."""

    search_results: List[MinimalAnswer]
    k: int


# ---------------------------------------------------------------------------
# Shared tokenizer
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Tokenize text, splitting snake_case and camelCase identifiers too.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercase tokens including sub-word splits.
    """
    tokens: List[str] = []
    for word in re.findall(r"\w+", text):
        tokens.append(word.lower())
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class RAGIndexer:
    """Index a codebase into a BM25 index with metadata."""

    def __init__(
        self,
        raw_data_path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
    ) -> None:
        """Initialise splitters for Python and Markdown files.

        Args:
            raw_data_path: Root directory of the corpus.
            max_chunk_size: Maximum characters per chunk.
        """
        self.raw_data_path = Path(raw_data_path)
        self.max_chunk_size = max_chunk_size

        self.py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=100,
            add_start_index=True,
        )
        self.md_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=100,
            add_start_index=True,
        )

    def retrieve_data(self, path: Path) -> str:
        """Read a file from disk, returning empty string on failure.

        Args:
            path: Path to the file.

        Returns:
            File contents as a string.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except OSError:
            return ""

    def ingest_corpus(self) -> List:
        """Read and chunk every .py and .md file in the corpus.

        Returns:
            List of LangChain Document objects with metadata.
        """
        files = (
            list(self.raw_data_path.rglob("*.py"))
            + list(self.raw_data_path.rglob("*.md"))
        )
        master_chunks: List = []
        for file_path in tqdm(files, desc="Ingesting files", unit="file"):
            content = self.retrieve_data(file_path)
            if not content.strip():
                continue
            splitter = (
                self.py_splitter if file_path.suffix == ".py" else self.md_splitter
            )
            docs = splitter.create_documents(
                texts=[content],
                metadatas=[{"file_path": str(file_path)}],
            )
            master_chunks.extend(docs)
        return master_chunks

    def build_metadata_and_corpus(self) -> Tuple[List[MinimalSource], List[str]]:
        """Build the metadata list and raw text corpus from chunked files.

        Returns:
            Tuple of (list of MinimalSource, list of chunk texts).
        """
        master_chunks = self.ingest_corpus()
        minimal_list: List[MinimalSource] = []
        raw_corpus: List[str] = []

        for chunk in master_chunks:
            start_idx = int(chunk.metadata["start_index"])
            text_len = len(chunk.page_content)
            source_obj = MinimalSource(
                file_path=chunk.metadata["file_path"],
                first_character_index=start_idx,
                last_character_index=start_idx + text_len,
            )
            minimal_list.append(source_obj)
            raw_corpus.append(chunk.page_content)

        return minimal_list, raw_corpus

    def build_and_save_index(
        self,
        sources: List[MinimalSource],
        corpus_texts: List[str],
        output_dir: str = "data/processed",
    ) -> None:
        """Tokenize the corpus, fit BM25, and persist index and metadata.

        Args:
            sources: Metadata for each chunk.
            corpus_texts: Raw text for each chunk.
            output_dir: Directory where the index and metadata are saved.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print("Tokenizing corpus chunks...")
        tokenized_corpus = [
            tokenize(doc)
            for doc in tqdm(corpus_texts, desc="Tokenizing", unit="chunk")
        ]

        print("Fitting BM25 model...")
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_file = out_path / "bm25.pkl"
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25, f)
        print(f"BM25 index saved to {bm25_file}")

        metadata_file = out_path / "metadata.json"
        metadata_payload = [source.model_dump() for source in sources]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)
        print(f"Metadata saved to {metadata_file}")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class RAGRetriever:
    """Retrieve relevant chunks from a pre-built BM25 index."""

    def __init__(self, processed_dir: str = "data/processed") -> None:
        """Load the BM25 index and metadata from disk.

        Args:
            processed_dir: Directory containing bm25.pkl and metadata.json.
        """
        self.processed_dir = Path(processed_dir)
        self.bm25 = self._load_bm25()
        self.metadata = self._load_metadata()

    def _load_bm25(self) -> BM25Okapi:
        """Load the pickled BM25 index.

        Returns:
            Loaded BM25Okapi instance.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        bm25_path = self.processed_dir / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"Index not found at {bm25_path}. Run index first."
            )
        try:
            with open(bm25_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as exc:
            raise RuntimeError(f"Failed to load BM25 index: {exc}") from exc

    def _load_metadata(self) -> List[Dict]:
        """Load the chunk metadata from JSON.

        Returns:
            List of metadata dicts.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
        """
        metadata_path = self.processed_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. Run index first."
            )
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _overlaps(a: MinimalSource, b: MinimalSource) -> bool:
        """Return True if two sources overlap by more than half.

        Args:
            a: First source.
            b: Second source.

        Returns:
            True when overlap exceeds 50 % of the shorter span.
        """
        if a.file_path != b.file_path:
            return False
        inter = min(a.last_character_index, b.last_character_index) - max(
            a.first_character_index, b.first_character_index
        )
        shorter = min(
            a.last_character_index - a.first_character_index,
            b.last_character_index - b.first_character_index,
        )
        return inter > 0.5 * shorter

    def search(self, query: str, top_k: int = 5) -> List[MinimalSource]:
        """Return the top_k sources ranked by BM25, skipping near-duplicates.

        Args:
            query: Natural-language question.
            top_k: Maximum number of results to return.

        Returns:
            List of MinimalSource objects ranked by relevance.
        """
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        results: List[MinimalSource] = []
        for idx in np.argsort(scores)[::-1]:
            if scores[idx] <= 0.0 or len(results) >= top_k:
                break
            cand = MinimalSource(**self.metadata[int(idx)])
            if not any(self._overlaps(cand, r) for r in results):
                results.append(cand)
        return results


# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------


class AnswerGenerator:
    """Generate grounded answers from retrieved snippets using Qwen3-0.6B."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 200,
        max_context_chars: int = 5000,
    ) -> None:
        """Load the tokenizer and model.

        Args:
            model_name: HuggingFace model identifier.
            max_new_tokens: Token budget for the generated answer.
            max_context_chars: Maximum characters of context fed to the model.
        """
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        )
        self.model.eval()

    @staticmethod
    def read_snippet(source: MinimalSource) -> str:
        """Re-read the text of a source from disk by character range.

        Args:
            source: Source location to read.

        Returns:
            The snippet text, or empty string on failure.
        """
        try:
            with open(source.file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            return ""
        return text[source.first_character_index: source.last_character_index]

    def generate(self, question: str, sources: List[MinimalSource]) -> str:
        """Answer the question using only the provided sources.

        Args:
            question: The question to answer.
            sources: Retrieved sources to use as context.

        Returns:
            Generated answer string.
        """
        context = "\n\n---\n\n".join(
            self.read_snippet(s) for s in sources[:3]
        )[: self.max_context_chars]
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the context. "
                    "Be concise. If the context is insufficient, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Compute character-range IoU between two sources in the same file.

    Args:
        a: First source.
        b: Second source.

    Returns:
        IoU score between 0.0 and 1.0.
    """
    if a.file_path != b.file_path:
        return 0.0
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    if inter <= 0:
        return 0.0
    union = max(a.last_character_index, b.last_character_index) - min(
        a.first_character_index, b.first_character_index
    )
    return inter / union


def is_found(gt: MinimalSource, retrieved: List[MinimalSource]) -> bool:
    """Return True if any retrieved source overlaps gt with IoU >= 0.05.

    Args:
        gt: Ground-truth source.
        retrieved: List of retrieved sources.

    Returns:
        True if at least one retrieved source matches.
    """
    return any(iou(gt, r) >= 0.05 for r in retrieved)


def recall_at_k(dataset: RagDataset, results: StudentSearchResults) -> float:
    """Compute recall@k over all answered questions in the dataset.

    Args:
        dataset: Ground-truth dataset.
        results: Student search results to evaluate.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    results_map = {
        r.question_id: r.retrieved_sources for r in results.search_results
    }
    total_gt = 0
    total_found = 0

    for q in dataset.rag_questions:
        if not isinstance(q, AnsweredQuestion):
            continue
        retrieved = results_map.get(q.question_id, [])
        for gt_source in q.sources:
            total_gt += 1
            if is_found(gt_source, retrieved):
                total_found += 1

    if total_gt == 0:
        return 0.0
    return total_found / total_gt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CLI:
    """Command-line interface for the RAG pipeline."""

    def index(
        self,
        raw_data_path: str = "data/raw/vllm-0.10.1",
        output_dir: str = "data/processed",
        max_chunk_size: int = 2000,
    ) -> None:
        """Ingest the corpus and build the BM25 index.

        Args:
            raw_data_path: Root directory of the corpus.
            output_dir: Where to save the index and metadata.
            max_chunk_size: Maximum characters per chunk.
        """
        indexer = RAGIndexer(
            raw_data_path=raw_data_path, max_chunk_size=max_chunk_size
        )
        sources, raw_corpus = indexer.build_metadata_and_corpus()
        indexer.build_and_save_index(sources, raw_corpus, output_dir=output_dir)

    def search(
        self,
        query: str,
        processed_dir: str = "data/processed",
        top_k: int = 5,
    ) -> None:
        """Search the index and print the top-k sources.

        Args:
            query: Natural-language question.
            processed_dir: Directory containing the index.
            top_k: Number of results to return.
        """
        retriever = RAGRetriever(processed_dir=processed_dir)
        results = retriever.search(query, top_k=top_k)
        for r in results:
            print(r.model_dump_json())

    def search_dataset(
        self,
        dataset_path: str,
        output_path: str,
        processed_dir: str = "data/processed",
        top_k: int = 5,
    ) -> None:
        """Run search over every question in a dataset file.

        Args:
            dataset_path: Path to the RagDataset JSON file.
            output_path: Where to write the StudentSearchResults JSON.
            processed_dir: Directory containing the index.
            top_k: Number of results per question.
        """
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        retriever = RAGRetriever(processed_dir=processed_dir)
        search_results: List[MinimalSearchResults] = []

        for q in tqdm(dataset.rag_questions, desc="Searching", unit="question"):
            sources = retriever.search(q.question, top_k=top_k)
            search_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=search_results, k=top_k)
        Path(output_path).write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Search results saved to {output_path}")

    def answer(
        self,
        query: str,
        processed_dir: str = "data/processed",
        top_k: int = 5,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Search and generate an answer for a single question.

        Args:
            query: Natural-language question.
            processed_dir: Directory containing the index.
            top_k: Number of sources to retrieve.
            model_name: HuggingFace model identifier.
        """
        retriever = RAGRetriever(processed_dir=processed_dir)
        generator = AnswerGenerator(model_name=model_name)
        sources = retriever.search(query, top_k=top_k)
        answer_text = generator.generate(query, sources)
        result = MinimalAnswer(
            question_id="cli",
            question=query,
            retrieved_sources=sources,
            answer=answer_text,
        )
        print(result.model_dump_json(indent=2))

    def answer_dataset(
        self,
        dataset_path: str,
        output_path: str,
        processed_dir: str = "data/processed",
        top_k: int = 5,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Search and generate answers for every question in a dataset file.

        Args:
            dataset_path: Path to the RagDataset JSON file.
            output_path: Where to write the StudentSearchResultsAndAnswer JSON.
            processed_dir: Directory containing the index.
            top_k: Number of sources per question.
            model_name: HuggingFace model identifier.
        """
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        retriever = RAGRetriever(processed_dir=processed_dir)
        generator = AnswerGenerator(model_name=model_name)
        answers: List[MinimalAnswer] = []

        for q in tqdm(dataset.rag_questions, desc="Answering", unit="question"):
            sources = retriever.search(q.question, top_k=top_k)
            answer_text = generator.generate(q.question, sources)
            answers.append(
                MinimalAnswer(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                    answer=answer_text,
                )
            )

        output = StudentSearchResultsAndAnswer(search_results=answers, k=top_k)
        Path(output_path).write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Answers saved to {output_path}")

    def evaluate(
        self,
        dataset_path: str,
        results_path: str,
        split: Optional[str] = None,
    ) -> None:
        """Print recall@k for a results file against a labelled dataset.

        Args:
            dataset_path: Path to the ground-truth RagDataset JSON.
            results_path: Path to the StudentSearchResults JSON.
            split: Optional filter: 'docs' (.md) or 'code' (.py).
        """
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        results = StudentSearchResults.model_validate_json(
            Path(results_path).read_text(encoding="utf-8")
        )

        if split == "docs":
            dataset = RagDataset(
                rag_questions=[
                    q for q in dataset.rag_questions
                    if isinstance(q, AnsweredQuestion)
                    and any(s.file_path.endswith(".md") for s in q.sources)
                ]
            )
        elif split == "code":
            dataset = RagDataset(
                rag_questions=[
                    q for q in dataset.rag_questions
                    if isinstance(q, AnsweredQuestion)
                    and any(s.file_path.endswith(".py") for s in q.sources)
                ]
            )

        score = recall_at_k(dataset, results)
        label = f" ({split})" if split else ""
        print(f"Recall@{results.k}{label}: {score:.2%}")


if __name__ == "__main__":
    fire.Fire(CLI)