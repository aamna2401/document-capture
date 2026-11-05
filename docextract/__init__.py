"""docextract: universal document extraction (PDF, images, tables) with cost-ordered engine cascade."""
from .schema import Block, Document, Page
from .pipeline import extract

__all__ = ["Block", "Document", "Page", "extract"]
