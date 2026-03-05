"""Comprehensive tests for BookGenerator including new features"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from book_generator import BookGenerator
from constants import AgentConstants, FileConstants


class TestBookGeneratorInit:
    """Test BookGenerator initialization"""

    def test_init_with_default_output_dir(self):
        """Test initialization with default output directory"""
        agents = {"user_proxy": Mock(), "writer": Mock()}
        outline = [{"chapter_number": 1, "title": "Test", "prompt": "Test prompt"}]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = BookGenerator(
                agents=agents,
                agent_config={"model": "test"},
                outline=outline,
                output_dir=tmpdir
            )
            assert generator.output_dir == tmpdir
            assert generator.chapters_memory == []
            assert generator.max_iterations == 3

    def test_init_creates_output_directory(self):
        """Test that initialization creates output directory"""
        agents = {"user_proxy": Mock(), "writer": Mock()}
        outline = [{"chapter_number": 1, "title": "Test", "prompt": "Test prompt"}]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "new_output")
            _ = BookGenerator(
                agents=agents,
                agent_config={"model": "test"},
                outline=outline,
                output_dir=output_dir
            )
            assert os.path.exists(output_dir)


class TestBookGeneratorVerification:
    """Test chapter verification methods"""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a BookGenerator instance for testing"""
        agents = {
            "user_proxy": Mock(),
            "writer": Mock(),
            "editor": Mock(),
            "memory_keeper": Mock()
        }
        outline = [
            {"chapter_number": 1, "title": "Chapter 1", "prompt": "Test prompt 1"},
            {"chapter_number": 2, "title": "Chapter 2", "prompt": "Test prompt 2"}
        ]
        return BookGenerator(
            agents=agents,
            agent_config={"model": "test"},
            outline=outline,
            output_dir=str(tmp_path)
        )

    def test_verify_chapter_content_valid(self, generator):
        """Test verifying valid chapter content"""
        content = "Chapter 1\n\nThis is valid chapter content with enough lines.\nLine 2.\nLine 3."
        assert generator._verify_chapter_content(content, 1)

    def test_verify_chapter_content_empty(self, generator):
        """Test verifying empty content"""
        assert not generator._verify_chapter_content("", 1)

    def test_verify_chapter_content_wrong_number(self, generator):
        """Test content with wrong chapter number"""
        content = "Chapter 2\n\nValid content here.\nMore content.\nEven more."
        assert not generator._verify_chapter_content(content, 1)

    def test_verify_chapter_content_only_metadata(self, generator):
        """Test content that's only metadata"""
        content = "Chapter 1"
        assert not generator._verify_chapter_content(content, 1)


class TestIntermediateDraftsFeature:
    """Test the new intermediate drafts saving feature"""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a BookGenerator instance for testing"""
        agents = {
            "user_proxy": Mock(),
            "writer": Mock(),
            "editor": Mock(),
            "memory_keeper": Mock()
        }
        outline = [{"chapter_number": 1, "title": "Chapter 1", "prompt": "Test prompt"}]
        return BookGenerator(
            agents=agents,
            agent_config={"model": "test"},
            outline=outline,
            output_dir=str(tmp_path)
        )

    def test_save_intermediate_drafts_creates_directory(self, generator):
        """Test that drafts directory is created"""
        messages = [
            {"sender": "writer", "content": f"{AgentConstants.SCENE_TAG}\nDraft content here.\n{AgentConstants.CHAPTER_END_TAG}"}
        ]
        generator._save_intermediate_drafts(1, messages)
        
        drafts_dir = Path(generator.output_dir) / FileConstants.DRAFTS_SUBDIR
        assert drafts_dir.exists()

    def test_save_intermediate_drafts_with_scene_tag(self, generator):
        """Test saving drafts from SCENE tag"""
        # Format: SCENE_TAG + content + CHAPTER_END_TAG
        messages = [
            {"sender": "writer", "name": "writer", "content": f"{AgentConstants.SCENE_TAG}\nThis is a writer draft with sufficient content to be saved. It needs to be at least one hundred characters long to pass the threshold test in the code.\n{AgentConstants.CHAPTER_END_TAG}"}
        ]
        generator._save_intermediate_drafts(1, messages)
        
        drafts_dir = Path(generator.output_dir) / FileConstants.DRAFTS_SUBDIR
        draft_files = list(drafts_dir.glob("*.md"))
        assert len(draft_files) == 1
        assert "writer_draft" in draft_files[0].name

    def test_save_intermediate_drafts_with_scene_final_tag(self, generator):
        """Test saving drafts from SCENE_FINAL tag"""
        messages = [
            {"sender": "writer_final", "content": f"{AgentConstants.SCENE_FINAL_TAG}\nThis is the final draft with enough content to be saved properly.\nIt needs to be quite long to pass the threshold.\n{AgentConstants.CHAPTER_END_TAG}"}
        ]
        generator._save_intermediate_drafts(1, messages)
        
        drafts_dir = Path(generator.output_dir) / FileConstants.DRAFTS_SUBDIR
        draft_files = list(drafts_dir.glob("*.md"))
        assert len(draft_files) == 1
        assert "final_draft" in draft_files[0].name

    def test_save_intermediate_drafts_with_chapter_tag(self, generator):
        """Test saving drafts from CHAPTER tag"""
        # Format: CHAPTER_START_TAG + content + CHAPTER_END_TAG
        messages = [
            {"sender": "writer", "name": "writer", "content": f"{AgentConstants.CHAPTER_START_TAG}\nThis is tagged chapter content with sufficient length to be saved as a draft. It must be over one hundred characters long to pass the threshold.\n{AgentConstants.CHAPTER_END_TAG}"}
        ]
        generator._save_intermediate_drafts(1, messages)
        
        drafts_dir = Path(generator.output_dir) / FileConstants.DRAFTS_SUBDIR
        draft_files = list(drafts_dir.glob("*.md"))
        assert len(draft_files) == 1
        assert "tagged_content" in draft_files[0].name

    def test_save_intermediate_drafts_skips_short_content(self, generator):
        """Test that short content is not saved"""
        messages = [
            {"sender": "writer", "content": f"{AgentConstants.SCENE_TAG} Short. {AgentConstants.CHAPTER_END_TAG}"}
        ]
        generator._save_intermediate_drafts(1, messages)
        
        drafts_dir = Path(generator.output_dir) / FileConstants.DRAFTS_SUBDIR
        draft_files = list(drafts_dir.glob("*.md"))
        assert len(draft_files) == 0

    def test_save_intermediate_drafts_multiple_drafts(self, generator):
        """Test saving multiple drafts from different messages"""
        long_content = "This is draft content with sufficient length to be saved. " * 5
        messages = [
            {"sender": "writer", "name": "writer", "content": f"{AgentConstants.SCENE_TAG}\n{long_content}\n{AgentConstants.CHAPTER_END_TAG}"},
            {"sender": "editor", "name": "editor", "content": f"{AgentConstants.SCENE_TAG}\n{long_content}\n{AgentConstants.CHAPTER_END_TAG}"},
            {"sender": "writer_final", "name": "writer_final", "content": f"{AgentConstants.SCENE_FINAL_TAG}\n{long_content}\n{AgentConstants.CHAPTER_END_TAG}"}
        ]
        generator._save_intermediate_drafts(1, messages)
        
        drafts_dir = Path(generator.output_dir) / FileConstants.DRAFTS_SUBDIR
        draft_files = list(drafts_dir.glob("*.md"))
        assert len(draft_files) == 3

    def test_save_intermediate_drafts_contains_metadata(self, generator):
        """Test that drafts contain metadata headers"""
        long_content = "Draft content here with enough length to be saved properly by the system. " * 3
        messages = [
            {"sender": "writer", "name": "writer", "content": f"{AgentConstants.SCENE_TAG}\n{long_content}\n{AgentConstants.CHAPTER_END_TAG}"}
        ]
        generator._save_intermediate_drafts(1, messages)
        
        drafts_dir = Path(generator.output_dir) / FileConstants.DRAFTS_SUBDIR
        draft_files = list(drafts_dir.glob("*.md"))
        assert len(draft_files) == 1
        
        content = draft_files[0].read_text()
        assert "<!-- Draft 1" in content
        assert "<!-- Message index:" in content
        assert "<!-- Sender:" in content


class TestCheckpointFeature:
    """Test checkpoint saving functionality"""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a BookGenerator instance for testing"""
        agents = {
            "user_proxy": Mock(),
            "writer": Mock()
        }
        outline = [{"chapter_number": 1, "title": "Chapter 1", "prompt": "Test prompt"}]
        return BookGenerator(
            agents=agents,
            agent_config={"model": "test"},
            outline=outline,
            output_dir=str(tmp_path)
        )

    def test_save_checkpoint_creates_directory(self, generator):
        """Test that checkpoint directory is created"""
        generator._save_checkpoint(1, "test_stage", {"key": "value"})
        
        checkpoint_dir = Path(generator.output_dir) / "checkpoints"
        assert checkpoint_dir.exists()

    def test_save_checkpoint_creates_file(self, generator):
        """Test that checkpoint file is created"""
        data = {"test": "data", "number": 42}
        generator._save_checkpoint(1, "content_extracted", data)
        
        checkpoint_dir = Path(generator.output_dir) / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        assert len(checkpoint_files) == 1
        
        # Verify content
        with open(checkpoint_files[0]) as f:
            saved = json.load(f)
        assert saved["chapter_number"] == 1
        assert saved["stage"] == "content_extracted"
        assert saved["data"] == data


class TestConversationLogFeature:
    """Test conversation log saving functionality"""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a BookGenerator instance for testing"""
        agents = {
            "user_proxy": Mock(),
            "writer": Mock()
        }
        outline = [{"chapter_number": 1, "title": "Chapter 1", "prompt": "Test prompt"}]
        return BookGenerator(
            agents=agents,
            agent_config={"model": "test"},
            outline=outline,
            output_dir=str(tmp_path)
        )

    def test_save_conversation_log_creates_directory(self, generator):
        """Test that conversation logs directory is created"""
        messages = [{"sender": "writer", "content": "Test message"}]
        generator._save_conversation_log(1, messages)
        
        log_dir = Path(generator.output_dir) / "conversation_logs"
        assert log_dir.exists()

    def test_save_conversation_log_creates_file(self, generator):
        """Test that conversation log file is created"""
        messages = [
            {"sender": "writer", "content": "Message 1"},
            {"sender": "editor", "content": "Message 2"}
        ]
        generator._save_conversation_log(1, messages)
        
        log_dir = Path(generator.output_dir) / "conversation_logs"
        log_files = list(log_dir.glob("*.json"))
        assert len(log_files) == 1
        
        # Verify content
        with open(log_files[0]) as f:
            saved = json.load(f)
        assert len(saved) == 2
        assert saved[0]["sender"] == "writer"


class TestBookStats:
    """Test get_book_stats functionality"""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a BookGenerator instance with some chapter files"""
        agents = {
            "user_proxy": Mock(),
            "writer": Mock()
        }
        outline = [
            {"chapter_number": 1, "title": "Chapter 1", "prompt": "Test prompt 1"},
            {"chapter_number": 2, "title": "Chapter 2", "prompt": "Test prompt 2"}
        ]
        return BookGenerator(
            agents=agents,
            agent_config={"model": "test"},
            outline=outline,
            output_dir=str(tmp_path)
        )

    def test_get_book_stats_empty(self, generator):
        """Test stats with no chapters"""
        stats = generator.get_book_stats()
        assert stats["total_chapters"] == 0
        assert stats["total_words"] == 0
        assert stats["chapters"] == []

    def test_get_book_stats_with_chapters(self, generator):
        """Test stats with existing chapters"""
        # Create some chapter files
        chapter1_path = Path(generator.output_dir) / "chapter_01.txt"
        chapter1_path.write_text("Chapter 1\n\nThis is chapter one content with words.")
        
        chapter2_path = Path(generator.output_dir) / "chapter_02.txt"
        chapter2_path.write_text("Chapter 2\n\nThis is chapter two content with more words here.")
        
        stats = generator.get_book_stats()
        assert stats["total_chapters"] == 2
        assert stats["total_words"] > 0
        assert len(stats["chapters"]) == 2

    def test_get_book_stats_skips_non_chapter_files(self, generator):
        """Test that non-chapter files are skipped"""
        # Create a chapter file and a non-chapter file
        chapter_path = Path(generator.output_dir) / "chapter_01.txt"
        chapter_path.write_text("Chapter 1\n\nContent here.")
        
        other_path = Path(generator.output_dir) / "readme.txt"
        other_path.write_text("This is not a chapter.")
        
        stats = generator.get_book_stats()
        assert stats["total_chapters"] == 1
        assert len(stats["chapters"]) == 1


class TestResumeCapability:
    """Test resume from checkpoint functionality"""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a BookGenerator instance for testing"""
        agents = {
            "user_proxy": Mock(),
            "writer": Mock()
        }
        outline = [
            {"chapter_number": 1, "title": "Chapter 1", "prompt": "Test prompt 1"},
            {"chapter_number": 2, "title": "Chapter 2", "prompt": "Test prompt 2"},
            {"chapter_number": 3, "title": "Chapter 3", "prompt": "Test prompt 3"}
        ]
        return BookGenerator(
            agents=agents,
            agent_config={"model": "test"},
            outline=outline,
            output_dir=str(tmp_path)
        )

    def test_find_resume_point_no_chapters(self, generator):
        """Test resume point when no chapters exist"""
        outline = [
            {"chapter_number": 1, "title": "Chapter 1", "prompt": "Test"},
            {"chapter_number": 2, "title": "Chapter 2", "prompt": "Test"}
        ]
        resume_point = generator._find_resume_point(outline)
        assert resume_point == 1

    def test_find_resume_point_with_completed_chapters(self, generator):
        """Test resume point when some chapters are complete"""
        # Create chapter 1 file
        chapter1_path = Path(generator.output_dir) / "chapter_01.txt"
        chapter1_path.write_text("Chapter 1\n\nValid content here.\nMore content.\nEven more.")
        
        outline = [
            {"chapter_number": 1, "title": "Chapter 1", "prompt": "Test"},
            {"chapter_number": 2, "title": "Chapter 2", "prompt": "Test"}
        ]
        resume_point = generator._find_resume_point(outline)
        assert resume_point == 2

    def test_find_resume_point_all_chapters_complete(self, generator):
        """Test resume point when all chapters are complete"""
        # Create chapter files
        chapter1_path = Path(generator.output_dir) / "chapter_01.txt"
        chapter1_path.write_text("Chapter 1\n\nValid content here.\nMore content.\nEven more.")
        
        chapter2_path = Path(generator.output_dir) / "chapter_02.txt"
        chapter2_path.write_text("Chapter 2\n\nValid content here.\nMore content.\nEven more.")
        
        chapter3_path = Path(generator.output_dir) / "chapter_03.txt"
        chapter3_path.write_text("Chapter 3\n\nValid content here.\nMore content.\nEven more.")
        
        outline = [
            {"chapter_number": 1, "title": "Chapter 1", "prompt": "Test"},
            {"chapter_number": 2, "title": "Chapter 2", "prompt": "Test"},
            {"chapter_number": 3, "title": "Chapter 3", "prompt": "Test"}
        ]
        resume_point = generator._find_resume_point(outline)
        assert resume_point == 4  # All chapters + 1


class TestContentExtraction:
    """Test content extraction methods"""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a BookGenerator instance for testing"""
        agents = {
            "user_proxy": Mock(),
            "writer": Mock(),
            "writer_final": Mock()
        }
        outline = [{"chapter_number": 1, "title": "Chapter 1", "prompt": "Test"}]
        return BookGenerator(
            agents=agents,
            agent_config={"model": "test"},
            outline=outline,
            output_dir=str(tmp_path)
        )

    def test_extract_final_scene_with_scene_final_tag(self, generator):
        """Test extracting content with SCENE_FINAL tag"""
        messages = [
            {"sender": "writer_final", "name": "writer_final", "content": f"{AgentConstants.SCENE_FINAL_TAG}\nFinal scene content here."}
        ]
        result = generator._extract_final_scene(messages)
        assert "Final scene content here." in result

    def test_extract_final_scene_with_scene_tag(self, generator):
        """Test extracting content with SCENE tag"""
        messages = [
            {"sender": "writer", "name": "writer", "content": f"{AgentConstants.SCENE_TAG}\nScene content here."}
        ]
        result = generator._extract_final_scene(messages)
        assert "Scene content here." in result

    def test_extract_final_scene_with_raw_content(self, generator):
        """Test extracting raw content without tags"""
        long_content = "This is a very long content that should be extracted. " * 50  # Make it > 500 chars
        messages = [
            {"sender": "writer", "name": "writer", "content": long_content}
        ]
        result = generator._extract_final_scene(messages)
        assert result is not None
        assert len(result) > 500

    def test_extract_final_scene_no_content(self, generator):
        """Test extracting when no content exists"""
        messages = [
            {"sender": "editor", "name": "editor", "content": "Just feedback, no scene."}
        ]
        result = generator._extract_final_scene(messages)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])