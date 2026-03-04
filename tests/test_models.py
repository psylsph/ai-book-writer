"""Comprehensive tests for data models"""
import pytest
from models import Chapter, Outline, ChapterContent, WorldElement


class TestChapter:
    """Test Chapter model validation"""

    def test_valid_chapter(self):
        chapter = Chapter(
            chapter_number=1,
            title="Test Chapter",
            prompt="This is a detailed test prompt with enough content"
        )
        assert chapter.chapter_number == 1
        assert chapter.title == "Test Chapter"

    def test_chapter_number_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            Chapter(chapter_number=0, title="Test", prompt="Valid prompt here")

    def test_title_cannot_be_empty(self):
        with pytest.raises(ValueError, match="empty"):
            Chapter(chapter_number=1, title="", prompt="Valid prompt")

    def test_title_stripped(self):
        chapter = Chapter(
            chapter_number=1,
            title="  Test Title  ",
            prompt="Valid prompt here"
        )
        assert chapter.title == "Test Title"

    def test_prompt_must_be_minimum_length(self):
        with pytest.raises(ValueError, match="at least 10"):
            Chapter(chapter_number=1, title="Test", prompt="Short")


class TestOutline:
    """Test Outline model validation"""

    def test_valid_outline(self):
        chapters = [
            Chapter(chapter_number=1, title="Ch1", prompt="Prompt 1" * 10),
            Chapter(chapter_number=2, title="Ch2", prompt="Prompt 2" * 10),
            Chapter(chapter_number=3, title="Ch3", prompt="Prompt 3" * 10)
        ]
        outline = Outline(chapters=chapters, total_chapters=3)
        assert len(outline.chapters) == 3

    def test_missing_chapter_numbers(self):
        chapters = [
            Chapter(chapter_number=1, title="Ch1", prompt="Prompt 1" * 10),
            Chapter(chapter_number=3, title="Ch3", prompt="Prompt 3" * 10)  # Missing #2
        ]
        with pytest.raises(ValueError, match="Missing"):
            Outline(chapters=chapters, total_chapters=3)

    def test_duplicate_chapter_numbers(self):
        chapters = [
            Chapter(chapter_number=1, title="Ch1a", prompt="Prompt 1a" * 10),
            Chapter(chapter_number=1, title="Ch1b", prompt="Prompt 1b" * 10)  # Duplicate
        ]
        with pytest.raises(ValueError, match="duplicate"):
            Outline(chapters=chapters, total_chapters=2)

    def test_chapters_sorted_by_number(self):
        chapters = [
            Chapter(chapter_number=3, title="Ch3", prompt="Prompt 3" * 10),
            Chapter(chapter_number=1, title="Ch1", prompt="Prompt 1" * 10),
            Chapter(chapter_number=2, title="Ch2", prompt="Prompt 2" * 10)
        ]
        outline = Outline(chapters=chapters, total_chapters=3)
        assert outline.chapters[0].chapter_number == 1
        assert outline.chapters[1].chapter_number == 2
        assert outline.chapters[2].chapter_number == 3


class TestChapterContent:
    """Test ChapterContent model"""

    def test_word_count_calculation(self):
        content = ChapterContent(
            chapter_number=1,
            title="Test",
            content="This is a test with exactly eight words in it"
        )
        assert content.word_count == 9

    def test_validate_length(self):
        content = ChapterContent(
            chapter_number=1,
            title="Test",
            content="word " * 5000
        )
        assert content.validate_length(min_words=5000) is True

    def test_validate_length_too_short(self):
        content = ChapterContent(
            chapter_number=1,
            title="Test",
            content="Too short"
        )
        assert content.validate_length(min_words=5000) is False


class TestWorldElement:
    """Test WorldElement model"""

    def test_valid_world_element(self):
        element = WorldElement(
            name="Dark Forest",
            description="A dense, mysterious forest with ancient trees",
            location_type="location",
            first_appearance=3
        )
        assert element.name == "Dark Forest"
        assert element.recurring is False

    def test_empty_name_not_allowed(self):
        with pytest.raises(ValueError):
            WorldElement(name="", description="Valid description")

    def test_short_description_not_allowed(self):
        with pytest.raises(ValueError):
            WorldElement(name="Test", description="Short")
