"""Custom exceptions for the book generation system"""
from typing import Optional


class BookGeneratorError(Exception):
    """Base exception for all book generator errors"""
    pass


class LLMError(BookGeneratorError):
    """Raised when LLM API calls fail"""
    def __init__(self, message: str, provider: str = "unknown", status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMTimeoutError(LLMError):
    """Raised when LLM API call times out"""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded"""
    def __init__(self, message: str, provider: str = "unknown", retry_after: Optional[int] = None):
        super().__init__(message, provider)
        self.retry_after = retry_after


class ParseError(BookGeneratorError):
    """Raised when parsing outline or chapter content fails"""
    def __init__(self, message: str, content: Optional[str] = None, field: Optional[str] = None):
        super().__init__(message)
        self.content = content
        self.field = field


class ValidationError(BookGeneratorError):
    """Raised when data validation fails"""
    def __init__(self, message: str, field: Optional[str] = None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value


class ChapterError(BookGeneratorError):
    """Raised when chapter generation fails"""
    def __init__(self, message: str, chapter_number: Optional[int] = None):
        super().__init__(message)
        self.chapter_number = chapter_number


class ChapterIncompleteError(ChapterError):
    """Raised when chapter generation is incomplete"""
    def __init__(self, message: str, chapter_number: int, missing_steps: Optional[list] = None):
        super().__init__(message, chapter_number)
        self.missing_steps = missing_steps or []


class ChapterTooShortError(ChapterError):
    """Raised when chapter content is too short"""
    def __init__(self, message: str, chapter_number: int, word_count: int, min_words: int):
        super().__init__(message, chapter_number)
        self.word_count = word_count
        self.min_words = min_words


class ConfigurationError(BookGeneratorError):
    """Raised when configuration is invalid"""
    pass


class FileOperationError(BookGeneratorError):
    """Raised when file operations fail"""
    def __init__(self, message: str, filepath: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(message)
        self.filepath = filepath
        self.operation = operation


class RetryExhaustedError(BookGeneratorError):
    """Raised when all retry attempts are exhausted"""
    def __init__(self, message: str, attempts: int, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
