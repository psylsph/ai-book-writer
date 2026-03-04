"""Constants for the book generation system - no magic numbers allowed!"""
import re


class ConfigConstants:
    """Configuration-related constants"""
    DEFAULT_MODEL = "Mistral-Nemo-Instruct-2407"
    DEFAULT_BASE_URL = "http://localhost:1234/v1"
    DEFAULT_API_KEY = "not-needed"
    DEFAULT_SEED = 42
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TIMEOUT = 600  # seconds


class FileConstants:
    """File path and naming constants"""
    OUTPUT_DIR = "book_output"
    OUTLINE_FILENAME = "outline.txt"
    CHAPTER_PREFIX = "chapter_"
    CHAPTER_EXTENSION = ".txt"
    BACKUP_EXTENSION = ".backup"


class ChapterConstants:
    """Chapter generation constants"""
    MIN_WORD_COUNT = 5000
    MIN_CONTENT_LENGTH = 100
    MIN_CONTENT_LINES = 3
    DEFAULT_NUM_CHAPTERS = 25
    CHAPTER_NUMBER_FORMAT = "02d"  # Format for zero-padding


class OutlineConstants:
    """Outline generation constants"""
    DEFAULT_NUM_CHAPTERS = 25
    MIN_EVENTS_PER_CHAPTER = 3
    OUTLINE_MAX_ROUNDS = 4
    

class GroupChatConstants:
    """Group chat configuration constants"""
    OUTLINE_MAX_ROUNDS = 4
    CHAPTER_MAX_ROUNDS = 5
    REPLY_MAX_ROUNDS = 3  # For retry attempts
    SPEAKER_SELECTION = "round_robin"


class RetryConstants:
    """Retry logic constants"""
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds
    MAX_DELAY = 60.0   # seconds
    EXPONENTIAL_BASE = 2.0


class AgentConstants:
    """Agent configuration constants"""
    MAX_ITERATIONS = 3
    MEMORY_UPDATE_TAG = "MEMORY UPDATE:"
    EVENT_TAG = "EVENT:"
    CHARACTER_TAG = "CHARACTER:"
    WORLD_TAG = "WORLD:"
    CONTINUITY_ALERT_TAG = "CONTINUITY ALERT:"
    SCENE_FINAL_TAG = "SCENE FINAL:"
    SCENE_TAG = "SCENE:"
    FEEDBACK_TAG = "FEEDBACK:"
    SUGGEST_TAG = "SUGGEST:"
    EDITED_SCENE_TAG = "EDITED_SCENE:"
    PLAN_TAG = "PLAN:"
    SETTING_TAG = "SETTING:"
    STORY_ARC_TAG = "STORY_ARC:"
    OUTLINE_START_TAG = "OUTLINE:"
    OUTLINE_END_TAG = "END OF OUTLINE"


class ValidationConstants:
    """Validation-related constants"""
    MIN_TITLE_LENGTH = 1
    MAX_TITLE_LENGTH = 200
    MIN_PROMPT_LENGTH = 10
    MAX_OUTLINE_CHAPTERS = 100


class LoggingConstants:
    """Logging configuration constants"""
    DEFAULT_LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RegexPatterns:
    """Pre-compiled regex patterns for parsing"""
    CHAPTER_NUMBER = re.compile(r'Chapter (\d+):', re.IGNORECASE)
    CHAPTER_SPLIT = re.compile(r'(?:Chapter|\*\*Chapter)\s+\d+:', re.IGNORECASE)
    
    # Title extraction
    TITLE = re.compile(r'\*?\*?Title:\*?\*?\s*(.+?)(?=\n|$)', re.IGNORECASE)
    CHAPTER_TITLE_ALT = re.compile(r'\*?\*?Chapter \d+:\s*(.+?)(?=\n|$)', re.IGNORECASE)
    
    # Component extraction
    KEY_EVENTS = re.compile(r'\*?\*?Key Events:\*?\*?\s*(.*?)(?=\*?\*?Character Developments:|$)', re.DOTALL | re.IGNORECASE)
    CHARACTER_DEVELOPMENTS = re.compile(r'\*?\*?Character Developments:\*?\*?\s*(.*?)(?=\*?\*?Setting:|$)', re.DOTALL | re.IGNORECASE)
    SETTING = re.compile(r'\*?\*?Setting:\*?\*?\s*(.*?)(?=\*?\*?Tone:|$)', re.DOTALL | re.IGNORECASE)
    TONE = re.compile(r'\*?\*?Tone:\*?\*?\s*(.*?)(?=\*?\*?Chapter \d+:|$)', re.DOTALL | re.IGNORECASE)
    
    # Content cleanup
    CHAPTER_REF = re.compile(r'\*?\s*\(Chapter \d+.*?\)')
    CHAPTER_HEADER = re.compile(r'\*?\s*Chapter \d+.*?\n', re.IGNORECASE)
    MARKDOWN_BOLD = re.compile(r'\*\*')
    
    # Bullet points
    BULLET_POINT = re.compile(r'-\s*(.+?)(?=\n|$)')
    
    # Scene tags
    SCENE_FINAL = re.compile(r'SCENE FINAL:', re.IGNORECASE)
    SCENE = re.compile(r'SCENE:', re.IGNORECASE)
    
    # Confirmation
    CONFIRMATION = re.compile(r'\*\*Confirmation:\*\*', re.IGNORECASE)