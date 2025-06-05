import logging
import sys
from pathlib import Path
from .settings import settings


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        # level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('chatbot')
    logger.setLevel(settings.LOG_LEVEL)
    logger.info("Logging system initialized")
    
    return logger


logger = setup_logging()        # Global instance