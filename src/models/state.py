from dataclasses import dataclass, field
from typing import List

from models.dto import ProcessingResult


@dataclass
class BatchState:
    total: int = 0
    processed: int = 0
    current_file: str = ''
    results: List[ProcessingResult] = field(default_factory=list)
