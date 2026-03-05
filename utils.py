"""Utility functions for the book generation system"""
import functools
import logging
import os
import random
import re
import shutil
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from exceptions import (
    ChapterTooShortError,
    FileOperationError,
    RetryExhaustedError,
)
from constants import AgentConstants, ChapterConstants, LoggingConstants, RegexPatterns, RetryConstants


# Setup logging
def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure and return the root logger"""
    logger = logging.getLogger("book_generator")
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            LoggingConstants.LOG_FORMAT,
            datefmt=LoggingConstants.DATE_FORMAT
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    log_level = level or LoggingConstants.DEFAULT_LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(f"book_generator.{name}")


# Retry decorator with exponential backoff
F = TypeVar('F', bound=Callable[..., Any])


def retry_with_backoff(
    max_retries: int = RetryConstants.MAX_RETRIES,
    base_delay: float = RetryConstants.BASE_DELAY,
    max_delay: float = RetryConstants.MAX_DELAY,
    exponential_base: float = RetryConstants.EXPONENTIAL_BASE,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[F], F]:
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        raise RetryExhaustedError(
                            message=f"Function {func.__name__} failed after {max_retries} attempts",
                            attempts=max_retries,
                            last_error=last_exception
                        ) from last_exception
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )
                    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                    total_delay = delay + jitter
                    
                    logger = get_logger("retry")
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed for {func.__name__}: {e}. "
                        f"Retrying in {total_delay:.2f}s..."
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(total_delay)
            
            raise RetryExhaustedError(
                message=f"Function {func.__name__} failed unexpectedly",
                attempts=max_retries,
                last_error=last_exception
            )
        
        return wrapper
    return decorator


# Content utilities

def count_words(text: str) -> int:
    """Count words in text, handling various whitespace"""
    if not text:
        return 0
    return len(text.split())


def validate_chapter_length(content: str, min_words: int = ChapterConstants.MIN_WORD_COUNT) -> bool:
    """Check if chapter meets minimum word count"""
    return count_words(content) >= min_words


def extract_content_between_tags(text: str, start_tag: str, end_tag: Optional[str] = None) -> Optional[str]:
    """Extract content between start and end tags"""
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    
    start_idx += len(start_tag)
    
    if end_tag:
        end_idx = text.find(end_tag, start_idx)
        if end_idx == -1:
            return None
        return text[start_idx:end_idx].strip()
    
    return text[start_idx:].strip()


def clean_chapter_content(content: str) -> str:
    """Clean chapter content by removing artifacts and formatting"""
    # Remove chapter number references in parentheses
    content = RegexPatterns.CHAPTER_REF.sub('', content)
    
    # Remove markdown bold markers
    content = RegexPatterns.MARKDOWN_BOLD.sub('', content)
    
    # Remove chapter header (first occurrence only)
    content = RegexPatterns.CHAPTER_HEADER.sub('', content, count=1)
    
    # Clean up whitespace
    lines = [line.strip() for line in content.split('\n')]
    content = '\n'.join(line for line in lines if line)
    
    return content.strip()


def get_sender_from_message(msg: dict) -> str:
    """Extract sender from message dict, handling different formats"""
    return msg.get("sender") or msg.get("name", "")


# Chapter sequence verification

def verify_chapter_sequence(chapters: list, expected_count: int) -> tuple[bool, list]:
    """
    Verify chapter numbering is sequential and complete
    
    Returns:
        (is_valid: bool, missing_numbers: list)
    """
    if len(chapters) != expected_count:
        return False, []
    
    expected_numbers = set(range(1, expected_count + 1))
    actual_numbers = set()
    
    for chapter in chapters:
        if isinstance(chapter, dict):
            num = chapter.get("chapter_number")
        else:
            num = getattr(chapter, "chapter_number", None)
        
        if num is not None:
            actual_numbers.add(num)
    
    missing = sorted(expected_numbers - actual_numbers)
    return len(missing) == 0, missing


# Formatting utilities

def format_chapter_title(chapter_num: int, title: str) -> str:
    """Format chapter title with consistent numbering"""
    return f"Chapter {chapter_num}: {title}"


def format_chapter_filename(chapter_num: int) -> str:
    """Generate standardized chapter filename"""
    from constants import FileConstants
    return f"{FileConstants.CHAPTER_PREFIX}{chapter_num:{ChapterConstants.CHAPTER_NUMBER_FORMAT}}{FileConstants.CHAPTER_EXTENSION}"


# Agent message tracking

def check_sequence_completion(messages: list) -> dict[str, bool]:
    """Check which sequence steps are complete in conversation"""
    sequence_complete = {
        'memory_update': False,
        'plan': False,
        'setting': False,
        'scene': False,
        'feedback': False,
        'scene_final': False,
        'confirmation': False
    }
    
    for msg in messages:
        content = msg.get("content", "")
        
        if AgentConstants.MEMORY_UPDATE_TAG in content:
            sequence_complete['memory_update'] = True
        if AgentConstants.PLAN_TAG in content:
            sequence_complete['plan'] = True
        if AgentConstants.SETTING_TAG in content:
            sequence_complete['setting'] = True
        if AgentConstants.SCENE_TAG in content:
            sequence_complete['scene'] = True
        if AgentConstants.FEEDBACK_TAG in content:
            sequence_complete['feedback'] = True
        if AgentConstants.SCENE_FINAL_TAG in content:
            sequence_complete['scene_final'] = True
        if AgentConstants.OUTLINE_END_TAG in content or "successfully" in content:
            sequence_complete['confirmation'] = True
    
    return sequence_complete


# String utilities

def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to max length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize string for use as filename"""
    # Remove any characters that aren't alphanumeric, underscore, hyphen, or dot
    sanitized = re.sub(r'[^\w\-\.]', '_', filename)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores and dots
    return sanitized.strip('_.')
