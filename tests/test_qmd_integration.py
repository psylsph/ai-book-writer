"""Tests for QMD (Query Markup Documents) integration module"""

import json
import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from qmd_integration import (
    QMDCLI,
    QMDConfig,
    QMDManager,
    QMDSearchResult,
    get_character_context,
    get_plot_context,
    search_book_content,
)


class TestQMDConfig:
    """Test QMD configuration"""

    def test_default_config(self):
        """Test QMDConfig default values"""
        config = QMDConfig()
        assert config.enabled is False
        assert config.collection_name == "book_chapters"
        assert config.kb_collection == "knowledge_base"
        assert config.auto_index is True
        assert config.index_drafts is False
        assert config.min_score == 0.3
        assert config.max_results == 5

    def test_config_from_env_all_enabled(self, monkeypatch):
        """Test QMDConfig from environment variables (all enabled)"""
        monkeypatch.setenv("QMD_ENABLED", "true")
        monkeypatch.setenv("QMD_COLLECTION_NAME", "my_book")
        monkeypatch.setenv("QMD_KB_COLLECTION", "my_kb")
        monkeypatch.setenv("QMD_AUTO_INDEX", "true")
        monkeypatch.setenv("QMD_INDEX_DRAFTS", "true")
        monkeypatch.setenv("QMD_MIN_SCORE", "0.5")
        monkeypatch.setenv("QMD_MAX_RESULTS", "10")

        config = QMDConfig.from_env()
        assert config.enabled is True
        assert config.collection_name == "my_book"
        assert config.kb_collection == "my_kb"
        assert config.auto_index is True
        assert config.index_drafts is True
        assert config.min_score == 0.5
        assert config.max_results == 10

    @pytest.mark.parametrize("value", ["1", "yes", "true", "TRUE"])
    def test_config_from_env_variants(self, monkeypatch, value):
        """Test QMDConfig with various boolean representations"""
        monkeypatch.setenv("QMD_ENABLED", value)
        config = QMDConfig.from_env()
        assert config.enabled is True, f"Failed for value: {value}"

    def test_config_from_env_disabled(self, monkeypatch):
        """Test QMDConfig when explicitly disabled"""
        monkeypatch.setenv("QMD_ENABLED", "false")
        config = QMDConfig.from_env()
        assert config.enabled is False


class TestQMDCLI:
    """Test QMD CLI wrapper"""

    @patch("qmd_integration.shutil.which")
    def test_cli_not_available(self, mock_which):
        """Test QMDCLI when qmd is not installed"""
        mock_which.return_value = None
        cli = QMDCLI()
        assert cli.is_available() is False

    @patch("qmd_integration.shutil.which")
    def test_cli_available(self, mock_which):
        """Test QMDCLI when qmd is installed"""
        mock_which.return_value = "/usr/bin/qmd"
        cli = QMDCLI()
        assert cli.is_available() is True
        assert cli._qmd_path == "/usr/bin/qmd"

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_collection_add_success(self, mock_exists, mock_run, mock_which):
        """Test successful collection add"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = QMDCLI()
        result = cli.collection_add("/path/to/chapters", name="test_collection")

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "collection" in call_args
        assert "add" in call_args
        assert "/path/to/chapters" in call_args

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_collection_add_with_name(self, mock_exists, mock_run, mock_which):
        """Test collection add with custom name"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = QMDCLI()
        cli.collection_add("/path/to/docs", name="my_collection")

        call_args = mock_run.call_args[0][0]
        assert "--name" in call_args
        assert "my_collection" in call_args

    @patch("qmd_integration.shutil.which")
    def test_collection_add_path_not_exists(self, mock_which):
        """Test collection add with non-existent path"""
        mock_which.return_value = "/usr/bin/qmd"
        cli = QMDCLI()
        result = cli.collection_add("/nonexistent/path")
        assert result is False

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_embed_success(self, mock_run, mock_which):
        """Test successful embed command"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(returncode=0, stdout="Embedding complete", stderr="")

        cli = QMDCLI()
        result = cli.embed()

        assert result is True
        mock_run.assert_called_once_with(
            ["/usr/bin/qmd", "embed"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300
        )

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_embed_failure(self, mock_run, mock_which):
        """Test embed command failure"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

        cli = QMDCLI()
        result = cli.embed()

        assert result is False

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_search_success(self, mock_run, mock_which):
        """Test successful search command"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_results = [
            {
                "doc_id": "chapter_01.txt",
                "title": "Chapter 1",
                "score": 0.85,
                "content": "Once upon a time...",
                "metadata": {"chapter": 1}
            }
        ]
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_results),
            stderr=""
        )

        cli = QMDCLI()
        results = cli.search("character motivation", max_results=5)

        assert len(results) == 1
        assert results[0].doc_id == "chapter_01.txt"
        assert results[0].score == 0.85
        mock_run.assert_called_once()

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_search_with_collection(self, mock_run, mock_which):
        """Test search with specific collection"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")

        cli = QMDCLI()
        cli.search("query", collection="my_collection")

        call_args = mock_run.call_args[0][0]
        assert "-c" in call_args
        assert "my_collection" in call_args

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_vsearch_success(self, mock_run, mock_which):
        """Test vector search"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_results = [
            {
                "doc_id": "chapter_02.txt",
                "title": "Chapter 2",
                "score": 0.92,
                "content": "The plot thickens...",
                "metadata": {}
            }
        ]
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(mock_results),
            stderr=""
        )

        cli = QMDCLI()
        results = cli.vsearch("what happens next", max_results=3)

        assert len(results) == 1
        assert results[0].doc_id == "chapter_02.txt"

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_query_hybrid_search(self, mock_run, mock_which):
        """Test hybrid query search"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")

        cli = QMDCLI()
        cli.query("plot twist", collection="book", max_results=10)

        call_args = mock_run.call_args[0][0]
        assert "query" in call_args
        assert "plot twist" in call_args
        assert "-n" in call_args
        assert "10" in call_args

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_get_document(self, mock_run, mock_which):
        """Test retrieving a specific document"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Full chapter content here",
            stderr=""
        )

        cli = QMDCLI()
        content = cli.get("chapter_05.txt", full_content=True)

        assert content == "Full chapter content here"
        call_args = mock_run.call_args[0][0]
        assert "get" in call_args
        assert "chapter_05.txt" in call_args
        assert "--full" in call_args

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_status(self, mock_run, mock_which):
        """Test status command"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Status: OK\nCollections: 2",
            stderr=""
        )

        cli = QMDCLI()
        status = cli.status()

        assert status["available"] is True
        assert "raw_output" in status

    @patch("qmd_integration.shutil.which")
    def test_cli_not_available_raises_error(self, mock_which):
        """Test that commands raise errors when CLI not available"""
        mock_which.return_value = None
        cli = QMDCLI()

        # search() catches the exception and returns empty list
        results = cli.search("query")
        assert results == []

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_command_timeout(self, mock_run, mock_which):
        """Test handling of command timeout"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.side_effect = subprocess.TimeoutExpired("qmd", 300)

        cli = QMDCLI()
        result = cli.embed()

        assert result is False

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_parse_search_results_empty(self, mock_run, mock_which):
        """Test parsing empty search results"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cli = QMDCLI()
        results = cli.search("query")

        assert results == []

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_parse_search_results_invalid_json(self, mock_run, mock_which):
        """Test parsing invalid JSON in search results"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not valid json",
            stderr=""
        )

        cli = QMDCLI()
        results = cli.search("query")

        assert results == []


class TestQMDManager:
    """Test QMD Manager"""

    @patch("qmd_integration.QMDCLI")
    def test_manager_initialization_disabled(self, mock_cli_class):
        """Test QMDManager when disabled"""
        config = QMDConfig(enabled=False)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli_class.return_value = mock_cli

        manager = QMDManager("output", config)

        assert manager.is_ready() is False
        mock_cli.collection_add.assert_not_called()

    @patch("qmd_integration.QMDCLI")
    def test_manager_initialization_enabled(self, mock_cli_class):
        """Test QMDManager when enabled and available"""
        config = QMDConfig(enabled=True, collection_name="test_book")
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli.collection_add.return_value = True
        mock_cli_class.return_value = mock_cli

        with patch("os.path.exists", return_value=True):
            manager = QMDManager("output", config)

        assert manager.is_ready() is True
        mock_cli.collection_add.assert_called_once_with("output", name="test_book")

    @patch("qmd_integration.QMDCLI")
    def test_manager_cli_not_available(self, mock_cli_class):
        """Test QMDManager when CLI not available"""
        config = QMDConfig(enabled=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = False
        mock_cli_class.return_value = mock_cli

        manager = QMDManager("output", config)

        assert manager.is_ready() is False

    @patch("qmd_integration.QMDCLI")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_index_chapter_success(self, mock_exists, mock_file, mock_cli_class):
        """Test successful chapter indexing"""
        config = QMDConfig(enabled=True, auto_index=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli.collection_add.return_value = True
        mock_cli.embed.return_value = True
        mock_cli_class.return_value = mock_cli

        manager = QMDManager("output", config)
        result = manager.index_chapter(5, "Chapter content here")

        assert result is True
        mock_cli.embed.assert_called_once()

    @patch("qmd_integration.QMDCLI")
    def test_index_chapter_not_ready(self, mock_cli_class):
        """Test indexing when manager not ready"""
        config = QMDConfig(enabled=False)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = False
        mock_cli_class.return_value = mock_cli

        manager = QMDManager("output", config)
        result = manager.index_chapter(1, "content")

        assert result is False

    @patch("qmd_integration.QMDCLI")
    def test_search_chapters(self, mock_cli_class):
        """Test searching chapters"""
        config = QMDConfig(enabled=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli.query.return_value = [
            QMDSearchResult(
                doc_id="chapter_01.txt",
                title="Chapter 1",
                score=0.85,
                content="Content...",
                metadata={}
            )
        ]
        mock_cli_class.return_value = mock_cli

        with patch("os.path.exists", return_value=True):
            manager = QMDManager("output", config)
            results = manager.search_chapters("query", max_results=3)

        assert len(results) == 1
        assert results[0].doc_id == "chapter_01.txt"
        mock_cli.query.assert_called_once_with(
            "query",
            collection="book_chapters",
            max_results=3,
            min_score=0.3
        )

    @patch("qmd_integration.QMDCLI")
    def test_search_characters(self, mock_cli_class):
        """Test character search"""
        config = QMDConfig(enabled=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli.query.return_value = []
        mock_cli_class.return_value = mock_cli

        with patch("os.path.exists", return_value=True):
            manager = QMDManager("output", config)
            manager.search_characters("Alice")

        call_args = mock_cli.query.call_args
        assert 'character "Alice"' in call_args[0][0]

    @patch("qmd_integration.QMDCLI")
    def test_get_continuity_context(self, mock_cli_class):
        """Test getting continuity context"""
        config = QMDConfig(enabled=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli.query.return_value = [
            QMDSearchResult(
                doc_id="chapter_01.txt",
                title="Chapter 1",
                score=0.9,
                content="Previous content",
                metadata={}
            )
        ]
        mock_cli_class.return_value = mock_cli

        with patch("os.path.exists", return_value=True):
            manager = QMDManager("output", config)
            context = manager.get_continuity_context(5, "character motivation")

        assert "Previous Context" in context
        assert "Chapter 1" in context

    @patch("qmd_integration.QMDCLI")
    def test_get_continuity_context_not_ready(self, mock_cli_class):
        """Test continuity context when not ready"""
        config = QMDConfig(enabled=False)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = False
        mock_cli_class.return_value = mock_cli

        manager = QMDManager("output", config)
        context = manager.get_continuity_context(5, "query")

        assert context == ""

    @patch("qmd_integration.QMDCLI")
    def test_format_search_results_empty(self, mock_cli_class):
        """Test formatting empty search results"""
        config = QMDConfig(enabled=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli_class.return_value = mock_cli

        with patch("os.path.exists", return_value=True):
            manager = QMDManager("output", config)
            formatted = manager.format_search_results_for_agent([], context_type="test")

        assert "No test information found" in formatted

    @patch("qmd_integration.QMDCLI")
    def test_format_search_results_with_data(self, mock_cli_class):
        """Test formatting search results with data"""
        config = QMDConfig(enabled=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli_class.return_value = mock_cli

        results = [
            QMDSearchResult(
                doc_id="ch1.txt",
                title="Chapter 1",
                score=0.85,
                content="A" * 1000,
                metadata={}
            ),
            QMDSearchResult(
                doc_id="ch2.txt",
                title="Chapter 2",
                score=0.75,
                content="B" * 1000,
                metadata={}
            )
        ]

        with patch("os.path.exists", return_value=True):
            manager = QMDManager("output", config)
            formatted = manager.format_search_results_for_agent(results, context_type="plot")

        assert "Relevant plot information" in formatted
        assert "Chapter 1" in formatted
        assert "Chapter 2" in formatted
        assert "(relevance: 0.85)" in formatted


class TestConvenienceFunctions:
    """Test convenience functions"""

    @patch("qmd_integration.QMDManager")
    def test_search_book_content(self, mock_manager_class):
        """Test search_book_content convenience function"""
        mock_manager = MagicMock()
        mock_manager.search_chapters.return_value = []
        mock_manager_class.return_value = mock_manager

        search_book_content("query", output_dir="test_output")

        mock_manager_class.assert_called_once_with("test_output")
        mock_manager.search_chapters.assert_called_once_with("query", max_results=5)

    @patch("qmd_integration.QMDManager")
    def test_get_character_context(self, mock_manager_class):
        """Test get_character_context convenience function"""
        mock_manager = MagicMock()
        mock_manager.search_characters.return_value = []
        mock_manager.format_search_results_for_agent.return_value = "Character info"
        mock_manager_class.return_value = mock_manager

        context = get_character_context("Alice", output_dir="test")

        assert context == "Character info"

    @patch("qmd_integration.QMDManager")
    def test_get_plot_context(self, mock_manager_class):
        """Test get_plot_context convenience function"""
        mock_manager = MagicMock()
        mock_manager.search_plot_points.return_value = []
        mock_manager.format_search_results_for_agent.return_value = "Plot info"
        mock_manager_class.return_value = mock_manager

        context = get_plot_context(output_dir="test", chapter_range="1-5")

        assert context == "Plot info"


class TestQMDIntegrationEdgeCases:
    """Test edge cases and error handling"""

    @patch("qmd_integration.QMDCLI")
    def test_index_chapter_exception_handling(self, mock_cli_class):
        """Test that indexing exceptions are handled gracefully"""
        config = QMDConfig(enabled=True, auto_index=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        mock_cli.embed.side_effect = Exception("Embed failed")
        mock_cli_class.return_value = mock_cli

        with patch("os.path.exists", return_value=True):
            manager = QMDManager("output", config)
            result = manager.index_chapter(1, "content")

        assert result is False

    @patch("qmd_integration.QMDCLI")
    def test_search_exception_handling(self, mock_cli_class):
        """Test that search exceptions are handled gracefully"""
        config = QMDConfig(enabled=True)
        mock_cli = MagicMock()
        mock_cli.is_available.return_value = True
        # Mock query to raise exception (this tests that exceptions propagate)
        mock_cli.query.side_effect = Exception("Search failed")
        mock_cli_class.return_value = mock_cli

        with patch("os.path.exists", return_value=True):
            manager = QMDManager("output", config)
            # The exception should propagate up (not be silently caught)
            with pytest.raises(Exception, match="Search failed"):
                manager.search_chapters("query")

    @patch("qmd_integration.shutil.which")
    @patch("subprocess.run")
    def test_called_process_error_handling(self, mock_run, mock_which):
        """Test handling of CalledProcessError"""
        mock_which.return_value = "/usr/bin/qmd"
        mock_run.side_effect = subprocess.CalledProcessError(1, "qmd")

        cli = QMDCLI()
        result = cli.embed()

        assert result is False

    def test_qmd_search_result_dataclass(self):
        """Test QMDSearchResult dataclass"""
        result = QMDSearchResult(
            doc_id="test.txt",
            title="Test",
            score=0.95,
            content="Content",
            metadata={"key": "value"}
        )

        assert result.doc_id == "test.txt"
        assert result.score == 0.95
        assert result.metadata == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])