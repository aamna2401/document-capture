"""Output schema. Every block keeps its page, bounding box, confidence and source for provenance."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

BlockType = Literal["text", "title", "table", "figure", "list", "header", "footer", "formula", "other"]
Source = Literal["text_layer", "ocr", "vlm"]


class Block(BaseModel):
    type: BlockType = "text"
    bbox: tuple[float, float, float, float]            # x0, y0, x1, y1 in PDF points (origin top-left)
    text: str = ""
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: Source = "text_layer"
    engine: str = ""
    table: Optional[list[list[str]]] = None            # rows of cell strings when type == "table"


class Page(BaseModel):
    number: int                                         # 1-based
    width: float
    height: float
    kind: Literal["born_digital", "scanned", "empty"] = "born_digital"
    blocks: list[Block] = []

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)


class Document(BaseModel):
    path: str
    sha256: str
    pages: list[Page] = []

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def to_markdown(self) -> str:
        out = []
        for p in self.pages:
            for b in p.blocks:
                if b.type == "table" and b.table:
                    head, *rows = b.table
                    out.append("| " + " | ".join(head) + " |")
                    out.append("|" + "---|" * len(head))
                    out += ["| " + " | ".join(r) + " |" for r in rows]
                elif b.type == "title":
                    out.append("# " + b.text)
                elif b.text:
                    out.append(b.text)
                out.append("")
        return "\n".join(out).strip() + "\n"
