import logging
import json
import time
import sys
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add extra properties passed in 'extra' dict
        for key, value in record.__dict__.items():
            if key not in {"args", "asctime", "created", "exc_info", "exc_text", 
                           "filename", "funcName", "levelname", "levelno", "lineno", 
                           "module", "msecs", "msg", "name", "pathname", "process", 
                           "processName", "relativeCreated", "stack_info", "thread", 
                           "threadName"}:
                log_data[key] = value
                
        return json.dumps(log_data)

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("clinical_copilot")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

logger = setup_logging()
