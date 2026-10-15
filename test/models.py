"""Pydantic data models for the RAG pipeline."""
from typing import List, Union

from pydantic import BaseModel


class MinimalSource(BaseModel):
    """A location in the corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without ground truth."""

    question_id: str
    question: str


class AnsweredQuestion(BaseModel):
    """A question with ground-truth sources and answer."""

    question_id: str
    question: str
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(BaseModel):
    """Retrieved sources plus the generated answer."""

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