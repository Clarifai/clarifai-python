from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ComputeConfig:
    """Compute requirements for an auto-generated pipeline step."""

    cpu_limit: Optional[str] = None
    cpu_memory: Optional[str] = None
    num_accelerators: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}