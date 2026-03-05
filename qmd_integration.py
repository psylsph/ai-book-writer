"""QMD (Query Markup Documents) integration for the book generator.

Provides search capabilities across generated chapters and knowledge base
documents using the QMD CLI tool.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from constants import FileConstants
from exceptions import ConfigurationError, FileOperationError
from utils import get_logger, retry_with_backoff

logger = get_logger("qmd_integration")


@dataclass
class QMDSearchResult:
    """Represents a single QMD search result"""

    doc_id: str
    title: str
    score: float
    content: str
    metadata: Dict[str, Any]


@dataclass
class QMDConfig:
    """Configuration for QMD integration"""

    enabled: bool = False
    collection_name: str = "book_chapters"
    kb_collection: str = "knowledge_base"
    auto_index: bool = True
    index_drafts: bool = False
    min_score: float = 0.3
    max_results: int = 5

    @classmethod
    def from_env(cls) -> "QMDConfig":
        """Create configuration from environment variables"""
        return cls(
            enabled=os.getenv("QMD_ENABLED", "false").lower() in ("true", "1", "yes"),
            collection_name=os.getenv("QMD_COLLECTION_NAME", "book_chapters"),
            kb_collection=os.getenv("QMD_KB_COLLECTION", "knowledge_base"),
            auto_index=os.getenv("QMD_AUTO_INDEX", "true").lower() in ("true", "1", "yes"),
            index_drafts=os.getenv("QMD_INDEX_DRAFTS", "false").lower() in ("true", "1", "yes"),
            min_score=float(os.getenv("QMD_MIN_SCORE", "0.3")),
            max_results=int(os.getenv("QMD_MAX_RESULTS", "5")),
        )


class QMDCLI:
    """Wrapper for QMD CLI commands"""

    def __init__(self) -> None:
        """Initialize QMD CLI wrapper"""
        self._qmd_path: Optional[str] = None
        self._check_installation()

    def _check_installation(self) -> None:
        """Check if QMD is installed and find its path"""
        self._qmd_path = shutil.which("qmd")
        if not self._qmd_path:
            logger.warning(
                "QMD not found in PATH. Install from https://github.com/tobi/qmd "
                "to enable search capabilities."
            )

    def is_available(self) -> bool:
        """Check if QMD CLI is available"""
        return self._qmd_path is not None

    def _run_command(
        self, args: List[str], check: bool = True, capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a QMD CLI command"""
        if not self._qmd_path:
            raise ConfigurationError("QMD CLI not available")

        cmd = [self._qmd_path] + args
        logger.debug(f"Running QMD command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=capture_output,
                text=True,
                timeout=300,  # 5 minute timeout for QMD operations
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"QMD command failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise FileOperationError(
                f"QMD command failed: {e}",
                operation="qmd_command",
                filepath=" ".join(cmd),
            ) from e
        except subprocess.TimeoutExpired as e:
            logger.error(f"QMD command timed out: {' '.join(cmd)}")
            raise FileOperationError(
                "QMD command timed out",
                operation="qmd_command",
                filepath=" ".join(cmd),
            ) from e

    def collection_add(
        self, path: str, name: Optional[str] = None, collection_type: str = "document"
    ) -> bool:
        """Add a directory to a QMD collection

        Args:
            path: Directory path to add
            name: Collection name (optional)
            collection_type: Type of collection (default: document)

        Returns:
            True if successful
        """
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            return False

        args = ["collection", "add", path]
        if name:
            args.extend(["--name", name])

        try:
            self._run_command(args, check=False)
            logger.info(f"Added {path} to QMD collection" + (f" '{name}'" if name else ""))
            return True
        except Exception as e:
            logger.error(f"Failed to add collection: {e}")
            return False

    def collection_remove(self, name: str) -> bool:
        """Remove a QMD collection"""
        try:
            self._run_command(["collection", "remove", name], check=False)
            logger.info(f"Removed QMD collection '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to remove collection: {e}")
            return False

    def embed(self) -> bool:
        """Generate embeddings for all indexed documents"""
        try:
            result = self._run_command(["embed"], check=False)
            if result.returncode == 0:
                logger.info("QMD embeddings generated successfully")
                return True
            else:
                logger.warning(f"QMD embed returned non-zero: {result.returncode}")
                return False
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return False

    def search(
        self,
        query: str,
        collection: Optional[str] = None,
        max_results: int = 5,
        min_score: float = 0.3,
        output_format: str = "json",
    ) -> List[QMDSearchResult]:
        """Search indexed documents using keyword search (BM25)

        Args:
            query: Search query
            collection: Collection name to search (optional)
            max_results: Maximum number of results
            min_score: Minimum relevance score
            output_format: Output format (json, files, etc.)

        Returns:
            List of search results
        """
        args = ["search", query, "-n", str(max_results), "--min-score", str(min_score)]

        if collection:
            args.extend(["-c", collection])

        if output_format == "json":
            args.append("--json")
        elif output_format == "files":
            args.append("--files")

        try:
            result = self._run_command(args)
            return self._parse_search_results(result.stdout)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def vsearch(
        self,
        query: str,
        collection: Optional[str] = None,
        max_results: int = 5,
        min_score: float = 0.3,
    ) -> List[QMDSearchResult]:
        """Vector semantic search across indexed documents

        Args:
            query: Search query (natural language)
            collection: Collection name to search (optional)
            max_results: Maximum number of results
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        args = ["vsearch", query, "-n", str(max_results), "--min-score", str(min_score)]

        if collection:
            args.extend(["-c", collection])

        try:
            result = self._run_command(args)
            return self._parse_search_results(result.stdout)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def query(
        self,
        query: str,
        collection: Optional[str] = None,
        max_results: int = 5,
        min_score: float = 0.3,
        include_all: bool = False,
    ) -> List[QMDSearchResult]:
        """Hybrid search with reranking (best quality)

        Combines BM25 keyword search, vector semantic search, and LLM reranking.

        Args:
            query: Search query
            collection: Collection name to search (optional)
            max_results: Maximum number of results
            min_score: Minimum relevance score
            include_all: Include all matches above threshold

        Returns:
            List of search results
        """
        args = ["query", query, "-n", str(max_results), "--min-score", str(min_score)]

        if collection:
            args.extend(["-c", collection])

        if include_all:
            args.append("--all")

        try:
            result = self._run_command(args)
            return self._parse_search_results(result.stdout)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get(self, doc_path: str, full_content: bool = False) -> Optional[str]:
        """Retrieve a specific document by path

        Args:
            doc_path: Document path or docid
            full_content: Get full document content

        Returns:
            Document content or None if not found
        """
        args = ["get", doc_path]
        if full_content:
            args.append("--full")

        try:
            result = self._run_command(args)
            return result.stdout.strip()
        except Exception as e:
            logger.error(f"Failed to get document {doc_path}: {e}")
            return None

    def status(self) -> Dict[str, Any]:
        """Get QMD index health and collection info"""
        try:
            result = self._run_command(["status"])
            # Parse status output (may need adjustment based on actual output format)
            return {"raw_output": result.stdout, "available": True}
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"available": False, "error": str(e)}

    def _parse_search_results(self, output: str) -> List[QMDSearchResult]:
        """Parse QMD search output into structured results

        QMD JSON output format is expected to be:
        [
          {
            "doc_id": "path/to/doc",
            "title": "Document Title",
            "score": 0.85,
            "content": "...",
            "metadata": {...}
          },
          ...
        ]
        """
        if not output.strip():
            return []

        try:
            data = json.loads(output)
            if not isinstance(data, list):
                logger.warning(f"Unexpected QMD output format: {type(data)}")
                return []

            results = []
            for item in data:
                if isinstance(item, dict):
                    results.append(
                        QMDSearchResult(
                            doc_id=item.get("doc_id", ""),
                            title=item.get("title", ""),
                            score=float(item.get("score", 0)),
                            content=item.get("content", ""),
                            metadata=item.get("metadata", {}),
                        )
                    )

            return results
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse QMD JSON output: {e}")
            logger.debug(f"Raw output: {output[:500]}")
            return []


class QMDManager:
    """High-level manager for QMD integration with the book generator"""

    def __init__(
        self,
        output_dir: str = FileConstants.OUTPUT_DIR,
        config: Optional[QMDConfig] = None,
    ):
        """Initialize QMD manager

        Args:
            output_dir: Directory containing book output
            config: QMD configuration (loads from env if not provided)
        """
        self.output_dir = output_dir
        self.config = config or QMDConfig.from_env()
        self.cli = QMDCLI()
        self._initialized = False

        if not self.cli.is_available():
            logger.warning("QMD CLI not available - search features disabled")
        elif not self.config.enabled:
            logger.info("QMD integration disabled in configuration")
        else:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize QMD collections"""
        if not self.cli.is_available() or not self.config.enabled:
            return

        try:
            # Add output directory to collection
            if os.path.exists(self.output_dir):
                self.cli.collection_add(
                    self.output_dir, name=self.config.collection_name
                )
                self._initialized = True
                logger.info(f"QMD manager initialized with collection '{self.config.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize QMD: {e}")
            self._initialized = False

    def is_ready(self) -> bool:
        """Check if QMD is ready for use"""
        return self._initialized and self.cli.is_available()

    @retry_with_backoff(max_retries=2)
    def index_chapter(self, chapter_number: int, content: str) -> bool:
        """Index a newly generated chapter

        Args:
            chapter_number: Chapter number
            content: Chapter content

        Returns:
            True if successfully indexed
        """
        if not self.is_ready() or not self.config.auto_index:
            return False

        try:
            # Create markdown version for better QMD indexing
            md_content = f"""# Chapter {chapter_number}

{content}
"""
            # Save to temporary markdown file
            md_path = os.path.join(
                self.output_dir, f"chapter_{chapter_number:02d}.md"
            )

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            # Re-index the collection
            success = self.cli.embed()

            if success:
                logger.info(f"Indexed Chapter {chapter_number} in QMD")
            else:
                logger.warning(f"Failed to index Chapter {chapter_number}")

            return success

        except Exception as e:
            logger.error(f"Error indexing chapter {chapter_number}: {e}")
            return False

    def search_chapters(
        self, query: str, max_results: Optional[int] = None
    ) -> List[QMDSearchResult]:
        """Search across all indexed chapters

        Args:
            query: Search query
            max_results: Maximum results to return (uses config default if None)

        Returns:
            List of search results
        """
        if not self.is_ready():
            logger.debug("QMD not ready for search")
            return []

        max_res = max_results or self.config.max_results

        # Use hybrid search for best results
        return self.cli.query(
            query,
            collection=self.config.collection_name,
            max_results=max_res,
            min_score=self.config.min_score,
        )

    def search_characters(self, character_name: str) -> List[QMDSearchResult]:
        """Search for information about a specific character

        Args:
            character_name: Name of the character to search for

        Returns:
            List of search results mentioning the character
        """
        query = f'character "{character_name}" appearance description background'
        return self.search_chapters(query, max_results=10)

    def search_plot_points(self, chapter_range: Optional[str] = None) -> List[QMDSearchResult]:
        """Search for major plot points in chapters

        Args:
            chapter_range: Optional range like "1-5" or "10-15"

        Returns:
            List of search results with plot events
        """
        query = "plot event happens occurred major turning point climax"
        if chapter_range:
            query += f" chapters {chapter_range}"
        return self.search_chapters(query, max_results=10)

    def get_chapter_summary(self, chapter_number: int) -> Optional[str]:
        """Get a summary of a specific chapter

        Args:
            chapter_number: Chapter number to summarize

        Returns:
            Chapter content or None if not found
        """
        if not self.is_ready():
            return None

        doc_path = f"{self.config.collection_name}/chapter_{chapter_number:02d}.md"
        return self.cli.get(doc_path, full_content=True)

    def get_continuity_context(
        self, chapter_number: int, query: str, max_results: int = 3
    ) -> str:
        """Get context from previous chapters for continuity

        Args:
            chapter_number: Current chapter number
            query: What to search for (e.g., "main character's motivation")
            max_results: Number of previous references to include

        Returns:
            Formatted context string for agent prompts
        """
        if not self.is_ready() or chapter_number <= 1:
            return ""

        results = self.search_chapters(query, max_results=max_results)

        if not results:
            return ""

        context_lines = ["Previous Context (from earlier chapters):"]
        for result in results:
            # Extract chapter number from doc_id if possible
            chapter_match = None
            if "chapter_" in result.doc_id:
                import re
                match = re.search(r"chapter_(\d+)", result.doc_id)
                if match:
                    chapter_match = int(match.group(1))

            # Only include chapters before current one
            if chapter_match and chapter_match >= chapter_number:
                continue

            context_lines.append(f"\nFrom {result.title} (relevance: {result.score:.2f}):")
            # Truncate content to reasonable length
            content_preview = result.content[:500] + "..." if len(result.content) > 500 else result.content
            context_lines.append(content_preview)

        return "\n".join(context_lines)

    def format_search_results_for_agent(
        self, results: List[QMDSearchResult], context_type: str = "general"
    ) -> str:
        """Format search results for inclusion in agent prompts

        Args:
            results: Search results to format
            context_type: Type of context (character, plot, setting, etc.)

        Returns:
            Formatted string for agent consumption
        """
        if not results:
            return f"No {context_type} information found in previous chapters."

        lines = [f"Relevant {context_type} information from previous chapters:"]

        for i, result in enumerate(results[:5], 1):
            lines.append(f"\n[{i}] {result.title} (relevance: {result.score:.2f})")
            # Include substantial content
            content = result.content[:800]
            if len(result.content) > 800:
                content += "..."
            lines.append(content)

        return "\n".join(lines)


# Convenience functions for direct use

def search_book_content(
    query: str,
    output_dir: str = FileConstants.OUTPUT_DIR,
    max_results: int = 5,
) -> List[QMDSearchResult]:
    """Search book content with a simple function interface

    Args:
        query: Search query
        output_dir: Book output directory
        max_results: Maximum results to return

    Returns:
        List of search results
    """
    manager = QMDManager(output_dir)
    return manager.search_chapters(query, max_results=max_results)


def get_character_context(
    character_name: str,
    output_dir: str = FileConstants.OUTPUT_DIR,
) -> str:
    """Get character context formatted for agents

    Args:
        character_name: Name of the character
        output_dir: Book output directory

    Returns:
        Formatted character context
    """
    manager = QMDManager(output_dir)
    results = manager.search_characters(character_name)
    return manager.format_search_results_for_agent(results, context_type="character")


def get_plot_context(
    output_dir: str = FileConstants.OUTPUT_DIR,
    chapter_range: Optional[str] = None,
) -> str:
    """Get plot context formatted for agents

    Args:
        output_dir: Book output directory
        chapter_range: Optional chapter range (e.g., "1-5")

    Returns:
        Formatted plot context
    """
    manager = QMDManager(output_dir)
    results = manager.search_plot_points(chapter_range)
    return manager.format_search_results_for_agent(results, context_type="plot")