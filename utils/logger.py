"""
Logging setup module. Configures root logger to write outputs to
console and output file at logs/rag.log.
"""

import logging
from pathlib import Path
from config.settings import settings

# Base directory setup
LOGS_DIR = Path(settings.CHROMA_PERSIST_DIRECTORY).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "rag.log"


def setup_logger(name: str = "rag_system") -> logging.Logger:
    """
    Sets up and configures the logger instance with console and file logs.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)

    # 1. Console Handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(log_level)
    c_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)

    # 2. File Handler
    try:
        f_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
        f_handler.setLevel(log_level)
        f_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)
    except Exception as e:
        logger.error(f"Failed to create file handler for logging at {LOG_FILE}: {str(e)}")

    logger.propagate = False
    return logger
