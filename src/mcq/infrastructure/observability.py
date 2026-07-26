"""Safe operational logging for the MCQ generation run."""

import logging
from pathlib import Path


def configure_logging(
    *, output_path: str, level: str = "INFO", log_path: str | None = None
) -> logging.Logger:
    path = Path(log_path) if log_path else Path(output_path).with_suffix(".run.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger("mcq.run")
    logger.info("Logging initialized; file=%s; level=%s", path, level.upper())
    return logger
