"""On-disk cache keyed by (file hash, page, engine, version), so reruns cost no compute or API calls."""
import json
from pathlib import Path

from .schema import Page

VERSION = "1"


class PageCache:
    def __init__(self, root="outputs/cache"):
        self.root = Path(root)

    def _path(self, sha: str, number: int, engine: str) -> Path:
        return self.root / sha[:2] / f"{sha}_{number}_{engine}_v{VERSION}.json"

    def get(self, sha, number, engine) -> Page | None:
        p = self._path(sha, number, engine)
        return Page.model_validate_json(p.read_text()) if p.exists() else None

    def put(self, sha, number, engine, page: Page):
        p = self._path(sha, number, engine)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(page.model_dump_json())
