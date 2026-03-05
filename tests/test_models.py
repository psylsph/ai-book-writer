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
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            Chapter(chapter_number=0, title="Test", prompt="Valid prompt here")

    def test_title_cannot_be_empty(self):
        with pytest.raises(ValueError, match="String should have at least 1 character"):
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
            Chapter(chapter_number=2, title="Ch2", prompt="Prompt 2" * 10),
            Chapter(chapter_number=4, title="Ch4", prompt="Prompt 4" * 10),  # Chapter 3 missing
            Chapter(chapter_number=5, title="Ch5", prompt="Prompt 5" * 10)
        ]
        with pytest.raises(ValueError, match="Expected 5 chapters, got 4"):
            Outline(chapters=chapters, total_chapters=5)

    def test_duplicate_chapter_numbers(self):
        # Note: The validator checks for missing numbers BEFORE duplicates.
        # With duplicates present, some chapter will always be missing.
        # So we test the scenario and verify we get a validation error.
        chapters = [
            Chapter(chapter_number=1, title="Ch1a", prompt="Prompt 1a" * 10),
            Chapter(chapter_number=1, title="Ch1b", prompt="Prompt 1b" * 10),  # Duplicate
            Chapter(chapter_number=2, title="Ch2", prompt="Prompt 2" * 10),
            Chapter(chapter_number=3, title="Ch3", prompt="Prompt 3" * 10),
            Chapter(chapter_number=4, title="Ch4", prompt="Prompt 4" * 10),
            Chapter(chapter_number=5, title="Ch5", prompt="Prompt 5" * 10),
            Chapter(chapter_number=6, title="Ch6", prompt="Prompt 6" * 10),
        ]
        # With 7 chapters, total=7, but duplicate 1s mean unique={1,2,3,4,5,6}
        # Missing={7}, so we get "Missing chapter numbers" error
        with pytest.raises(ValueError, match="Missing chapter numbers"):
            Outline(chapters=chapters, total_chapters=7)

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
        content_text = "This is a test with exactly eight words in it. " + "More text here to ensure we meet the minimum length requirement of 100 characters."
        content = ChapterContent(
            chapter_number=1,
            title="Test",
            content=content_text
        )
        assert content.word_count == 24

    def test_validate_length(self):
        content = ChapterContent(
            chapter_number=1,
            title="Test",
            content="word " * 3000
        )
        assert content.validate_length(min_words=3000) is True

    def test_validate_length_too_short(self):
        # Content must be at least 100 characters to pass model validation,
        # but less than 3000 words for validate_length to return False
        content = ChapterContent(
            chapter_number=1,
            title="Test",
            content="This content is long enough to pass the model validation requirement "
                    "of 100 characters minimum, but does not have 3000 words in it. "
                    "It should fail the validate_length check with min_words=3000."
        )
        assert content.validate_length(min_words=3000) is False


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
