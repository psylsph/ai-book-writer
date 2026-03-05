"""Tests for the book generator"""
import pytest
from models import Chapter, Outline
from constants import ChapterConstants
from utils import count_words, validate_chapter_length, clean_chapter_content


class TestChapterModel:
    """Test Chapter pydantic model"""

    def test_valid_chapter(self):
        """Test creating a valid chapter"""
        chapter = Chapter(
            chapter_number=1,
            title="Test Chapter",
            prompt="This is a test prompt that is long enough"
        )
        assert chapter.chapter_number == 1
        assert chapter.title == "Test Chapter"

    def test_chapter_number_validation(self):
        """Test chapter number validation"""
        with pytest.raises(ValueError):
            Chapter(chapter_number=0, title="Test", prompt="Valid prompt here")

    def test_title_validation(self):
        """Test title validation"""
        with pytest.raises(ValueError):
            Chapter(chapter_number=1, title="", prompt="Valid prompt here")

    def test_prompt_validation(self):
        """Test prompt validation"""
        with pytest.raises(ValueError):
            Chapter(chapter_number=1, title="Test", prompt="Short")


class TestOutlineModel:
    """Test Outline pydantic model"""

    def test_valid_outline(self):
        """Test creating a valid outline"""
        chapters = [
            Chapter(chapter_number=1, title="Chapter 1", prompt="Prompt 1" * 10),
            Chapter(chapter_number=2, title="Chapter 2", prompt="Prompt 2" * 10)
        ]
        outline = Outline(chapters=chapters, total_chapters=2)
        assert len(outline.chapters) == 2

    def test_missing_chapters(self):
        """Test outline with missing chapters"""
        chapters = [Chapter(chapter_number=1, title="Chapter 1", prompt="Prompt 1" * 10)]
        with pytest.raises(ValueError):
            Outline(chapters=chapters, total_chapters=2)


class TestUtils:
    """Test utility functions"""

    def test_count_words(self):
        """Test word counting"""
        text = "This is a test with five words"
        assert count_words(text) == 6

    def test_count_words_empty(self):
        """Test word counting with empty string"""
        assert count_words("") == 0

    def test_valid_chapter_length(self):
        """Test chapter length validation"""
        text = "word " * ChapterConstants.MIN_WORD_COUNT
        assert validate_chapter_length(text)

    def test_invalid_chapter_length(self):
        """Test chapter length validation with short text"""
        text = "This is too short"
        assert not validate_chapter_length(text)

    def test_clean_chapter_content(self):
        """Test content cleaning"""
        content = "**Chapter 1: Title**\n\nSome content here"
        cleaned = clean_chapter_content(content)
        assert "Chapter 1: Title" not in cleaned
        assert "Some content here" in cleaned


if __name__ == "__main__":
    pytest.main([__file__])