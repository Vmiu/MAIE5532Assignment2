"""
Logger utility functions for creating isolated loggers that don't interfere with each other.
"""

import logging
import os
from typing import Optional


def get_logger(name: str, log_file: str, log_level: int = logging.INFO, 
               console_output: bool = True, file_mode: str = 'w') -> logging.Logger:
    """
    Create a completely isolated logger that won't affect other log files.
    
    Args:
        name: Unique name for the logger (e.g., 'part1_baseline', 'part2_cloud_optimization')
        log_file: Path to the log file (directory will be created if it doesn't exist)
        log_level: Logging level (default: logging.INFO)
        console_output: Whether to output to console (default: True)
        file_mode: File mode for log file ('w' for write, 'a' for append, default: 'w')
    
    Returns:
        logging.Logger: Isolated logger instance
    """
    
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique logger with the specified name
    logger = logging.getLogger(f"isolated_{name}")
    logger.setLevel(log_level)
    
    # Clear any existing handlers to ensure complete isolation
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # CRITICAL: Prevent propagation to root logger to avoid affecting other loggers
    logger.propagate = False
    
    # Prevent the logger from being garbage collected
    logger.disabled = False
    
    return logger


def get_simple_logger(name: str, log_file: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Simplified version of get_logger with sensible defaults.
    
    Args:
        name: Unique name for the logger
        log_file: Path to the log file
        log_level: Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Isolated logger instance
    """
    return get_logger(name, log_file, log_level, console_output=True, file_mode='w')


def cleanup_logger(logger: logging.Logger) -> None:
    """
    Properly cleanup a logger by closing all handlers.
    
    Args:
        logger: Logger instance to cleanup
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.disabled = True


# Example usage and testing
if __name__ == "__main__":
    # Test the logger isolation
    logger1 = get_logger("test_part1", "test1_logs/part.log")
    logger2 = get_logger("test_part2", "test2_logs/part.log")
    
    logger1.info("This should only appear in part1.log")
    logger2.info("This should only appear in part2.log")
    
    print("Logger isolation test completed. Check test_logs/ directory for log files.")
