"""Pytest fixtures and configuration"""
import pytest
from models import Chapter


@pytest.fixture
def sample_chapter():
    """Provide a sample chapter for tests"""
    return Chapter(
        chapter_number=1,
        title="Test Chapter",
        prompt="This is a detailed test prompt with enough content to pass validation"
    )


@pytest.fixture
def sample_chapters():
    """Provide multiple sample chapters"""
    return [
        Chapter(
            chapter_number=i,
            title=f"Chapter {i}",
            prompt=f"This is the prompt for chapter {i} with enough content"
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def mock_agent_config():
    """Provide a mock agent configuration"""
    return {
        "seed": 42,
        "temperature": 0.7,
        "timeout": 600,
        "config_list": [{
            "model": "test-model",
            "base_url": "http://localhost:1234/v1",
            "api_key": "not-needed"
        }]
    }


@pytest.fixture
def mock_outline():
    """Provide a mock outline structure"""
    return [
        {
            "chapter_number": 1,
            "title": "Chapter One",
            "prompt": "Key Events:\n- Event 1\n- Event 2\n- Event 3\n\nCharacter Developments: Character growth\nSetting: Office\nTone: Tense"
        },
        {
            "chapter_number": 2,
            "title": "Chapter Two",
            "prompt": "Key Events:\n- Event A\n- Event B\n- Event C\n\nCharacter Developments: New relationships\nSetting: Home\nTone: Reflective"
        }
    ]
