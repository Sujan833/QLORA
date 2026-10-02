"""
Checkpoint Manager for Universal QLoRA Fine-Tuner.
"""

from pathlib import Path
from typing import List, Optional

from qlora_engine.utils.logging import get_logger

logger = get_logger("qlora_engine.checkpointing")


class CheckpointManager:
    """
    Manages output directory structure, checkpoint listing, and latest checkpoint discovery.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> List[Path]:
        """
        Lists all available step or epoch checkpoint directories sorted by modification time.
        """
        if not self.checkpoints_dir.exists():
            return []
        checkpoints = [
            p for p in self.checkpoints_dir.glob("checkpoint-*") if p.is_dir()
        ]
        return sorted(checkpoints, key=lambda x: x.stat().st_mtime)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Returns the path to the most recent checkpoint or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
