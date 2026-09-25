"""Uniform, minimal logging setup used across all pipeline stages."""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Don't propagate to the root logger: run_pipeline stages import many
        # modules, and if anything (a notebook, a future file handler)
        # configures the root logger, every record would print twice.
        logger.propagate = False
    return logger
