"""Main class for generating books using AutoGen with improved error handling

Supports both legacy AutoGen 0.2 and AutoGen 2.0 (0.4+) APIs.
AutoGen 2.0 uses async team execution with RoundRobinGroupChat and termination conditions.
"""
import asyncio
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

# AutoGen 2.0 imports
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.agents import AssistantAgent
AUTOGEN_2_AVAILABLE = True

from constants import (
    AgentConstants,
    ChapterConstants,
    ConfigConstants,
    FileConstants,
    GroupChatConstants,
    RegexPatterns,
)
from exceptions import (
    ChapterError,
    ChapterTooShortError,
    FileOperationError,
    RetryExhaustedError,
)
from qmd_integration import QMDConfig, QMDManager
from utils import (
    check_sequence_completion,
    clean_chapter_content,
    count_words,
    extract_content_between_tags,
    format_chapter_filename,
    get_logger,
    get_sender_from_message,
    retry_with_backoff,
)

logger = get_logger("book_generator")


class BookGenerator:
    """Generates book chapters using multi-agent collaboration with validation

    Supports both legacy AutoGen 0.2 and AutoGen 2.0 (0.4+) APIs.
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        agent_config: Dict[str, Any],
        outline: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        qmd_config: Optional[QMDConfig] = None,
        use_autogen2: bool = True,
    ):
        """Initialize with outline to maintain chapter count context

        Args:
            agents: Dictionary of AutoGen agents
            agent_config: Configuration for agents (dict for legacy, model_client for 2.0)
            outline: Book outline with chapter specifications
            output_dir: Directory for output files
            qmd_config: QMD configuration for search integration (optional)
            use_autogen2: Use AutoGen 2.0 API if True, legacy API if False
        """
        self.agents = agents
        self.agent_config = agent_config
        self.output_dir = output_dir or FileConstants.OUTPUT_DIR
        self.chapters_memory: List[str] = []
        self.max_iterations = 3
        self.outline = outline
        self.use_autogen2 = use_autogen2 and AUTOGEN_2_AVAILABLE

        # Initialize QMD manager for search capabilities
        self.qmd_manager: Optional[QMDManager] = None
        if qmd_config is None:
            qmd_config = QMDConfig.from_env()

        if qmd_config.enabled:
            self.qmd_manager = QMDManager(self.output_dir, qmd_config)
            if self.qmd_manager.is_ready():
                logger.info("QMD search integration enabled")
            else:
                logger.warning("QMD enabled but not available - search features disabled")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"BookGenerator initialized (AutoGen 2.0: {self.use_autogen2})")

    def _find_resume_point(self, outline: List[Dict[str, Any]]) -> int:
        """Find which chapter to resume from based on existing files"""
        for chapter in sorted(outline, key=lambda x: x["chapter_number"]):
            chapter_number = chapter["chapter_number"]
            chapter_file = os.path.join(
                self.output_dir, format_chapter_filename(chapter_number)
            )

            if not os.path.exists(chapter_file):
                logger.info(f"Resuming from Chapter {chapter_number}")
                return chapter_number

            # Check if file is valid
            try:
                with open(chapter_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if not self._verify_chapter_content(content, chapter_number):
                        logger.info(f"Chapter {chapter_number} exists but is invalid, regenerating")
                        return chapter_number
            except Exception:
                return chapter_number

        logger.info("All chapters already generated")
        return len(outline) + 1

    def generate_book(self, outline: List[Dict[str, Any]], resume: bool = True) -> None:
        """Generate the book with strict chapter sequencing

        Args:
            outline: List of chapter specifications
            resume: If True, skip chapters that already exist and are valid
        """
        logger.info("Starting Book Generation")
        logger.info(f"Using AutoGen 2.0: {self.use_autogen2}")
        logger.info(f"Total chapters: {len(outline)}")

        # Sort outline by chapter number
        sorted_outline = sorted(outline, key=lambda x: x["chapter_number"])

        # Find resume point if requested
        start_chapter = 1
        if resume:
            start_chapter = self._find_resume_point(outline)
            if start_chapter > len(outline):
                logger.info("All chapters already generated and valid")
                return

        for chapter in sorted_outline:
            chapter_number = chapter["chapter_number"]

            # Skip chapters before resume point
            if chapter_number < start_chapter:
                logger.info(f"Skipping Chapter {chapter_number} (already exists)")
                continue
            chapter_number = chapter["chapter_number"]

            # Verify previous chapter exists and is valid
            if chapter_number > 1:
                if not self._verify_previous_chapter(chapter_number - 1):
                    logger.error(f"Previous chapter {chapter_number - 1} invalid. Stopping.")
                    break

            # Generate current chapter
            logger.info(f"{'='*20} Chapter {chapter_number} {'='*20}")

            try:
                self.generate_chapter(chapter_number, chapter["prompt"])
            except ChapterError as e:
                logger.error(f"Chapter {chapter_number} generation failed: {e}")
                self._save_checkpoint(chapter_number, f"failed_{e.__class__.__name__}", {"error": str(e)})
                raise

            # Verify current chapter
            if not self._verify_chapter_file(chapter_number):
                logger.error(f"Failed to generate chapter {chapter_number}")
                self._save_checkpoint(chapter_number, "verify_failed", {})
                break

            logger.info(f"✓ Chapter {chapter_number} complete")
            time.sleep(5)

        logger.info("Book generation complete")

    def _verify_previous_chapter(self, chapter_number: int) -> bool:
        """Verify previous chapter exists and is valid"""
        chapter_file = os.path.join(
            self.output_dir, format_chapter_filename(chapter_number)
        )

        if not os.path.exists(chapter_file):
            logger.error(f"Previous chapter {chapter_number} file not found: {chapter_file}")
            return False

        try:
            with open(chapter_file, "r", encoding="utf-8") as f:
                content = f.read()
                if not self._verify_chapter_content(content, chapter_number):
                    logger.error(f"Previous chapter {chapter_number} content invalid")
                    return False
        except Exception as e:
            logger.error(f"Error reading chapter {chapter_number}: {e}")
            return False

        return True

    def _verify_chapter_file(self, chapter_number: int) -> bool:
        """Verify chapter file exists and has valid content"""
        chapter_file = os.path.join(
            self.output_dir, format_chapter_filename(chapter_number)
        )

        if not os.path.exists(chapter_file):
            logger.error(f"Chapter file not created: {chapter_file}")
            return False

        try:
            with open(chapter_file, "r", encoding="utf-8") as f:
                content = f.read()
                return self._verify_chapter_content(content, chapter_number)
        except Exception as e:
            logger.error(f"Error verifying chapter {chapter_number}: {e}")
            return False

    def _verify_chapter_content(self, content: str, chapter_number: int) -> bool:
        """Verify chapter content is valid"""
        if not content:
            return False

        # Check for chapter header
        if f"Chapter {chapter_number}" not in content:
            return False

        # Ensure content isn't just metadata
        lines = content.split("\n")
        content_lines = [line for line in lines if line.strip()]

        return len(content_lines) >= ChapterConstants.MIN_CONTENT_LINES

    @retry_with_backoff(max_retries=3)
    def generate_chapter(self, chapter_number: int, prompt: str) -> None:
        """Generate a single chapter with completion verification"""
        logger.info(f"Generating Chapter {chapter_number}...")

        messages: List[Dict[str, Any]] = []

        try:
            messages = self._generate_chapter_autogen2(chapter_number, prompt)

            # Check if chapter generation sequence is complete
            # If complete, discard any messages that came after completion
            completion_index = self._find_chapter_completion_index(messages)
            if completion_index is not None:
                logger.info(f"Chapter {chapter_number} completed at message index {completion_index}")
                # Only keep messages up to and including completion
                messages = messages[:completion_index + 1]

            # Check completion status
            sequence_complete = check_sequence_completion(messages)
            missing_steps = [step for step, complete in sequence_complete.items() if not complete]

            if missing_steps:
                logger.warning(
                    f"Chapter {chapter_number} has incomplete steps: {missing_steps}. "
                    f"Attempting to process available content..."
                )

            # Always try to process results, even if incomplete
            self._process_chapter_results(chapter_number, messages)

            # Check if file was created
            chapter_file = os.path.join(
                self.output_dir, format_chapter_filename(chapter_number)
            )

            if not os.path.exists(chapter_file):
                logger.error(f"Chapter file not created: {chapter_file}")
                # Only use emergency generation if no file was created at all
                raise FileOperationError(
                    f"Chapter {chapter_number} file not created",
                    filepath=chapter_file,
                    operation="create"
                )

            # Verify the saved content
            if not self._verify_chapter_file(chapter_number):
                logger.error(f"Chapter {chapter_number} file verification failed")
                raise ChapterError(
                    f"Chapter {chapter_number} verification failed",
                    chapter_number=chapter_number
                )

            # Send confirmation to complete the sequence
            if not self.use_autogen2:
                # Legacy mode - send confirmation through agent
                completion_msg = f"Chapter {chapter_number} is complete. Proceed with next chapter."
                # Note: In AutoGen 2.0, this is handled differently
                logger.info(f"Chapter {chapter_number} completion confirmed")

            logger.info(f"✓ Chapter {chapter_number} generated successfully")

        except RetryExhaustedError:
            logger.error(f"All retry attempts exhausted for chapter {chapter_number}")
            raise
        except (FileOperationError, ChapterError) as e:
            # These errors mean we should try emergency generation
            logger.error(f"Critical error in chapter {chapter_number}: {e}")
            self._save_checkpoint(chapter_number, "critical_error", {"error": str(e), "messages": messages})
            logger.info(f"Attempting emergency generation for chapter {chapter_number}...")
            self._handle_chapter_generation_failure(chapter_number, prompt)
            logger.info(f"Emergency generation completed for chapter {chapter_number}")
        except Exception as e:
            # Log unexpected errors but don't necessarily trigger emergency mode
            logger.error(f"Unexpected error in chapter {chapter_number}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._save_checkpoint(chapter_number, "unexpected_error", {"error": str(e), "messages": messages})
            # Check if we have a file before deciding to use emergency generation
            chapter_file = os.path.join(
                self.output_dir, format_chapter_filename(chapter_number)
            )
            if not os.path.exists(chapter_file):
                logger.warning("No chapter file exists, attempting emergency generation...")
                self._handle_chapter_generation_failure(chapter_number, prompt)
            else:
                logger.info("Chapter file exists despite error, continuing...")

    def _generate_chapter_autogen2(
        self, chapter_number: int, prompt: str
    ) -> List[Dict[str, Any]]:
        """Generate chapter using AutoGen 2.0"""
        if not AUTOGEN_2_AVAILABLE:
            raise ImportError(
                "AutoGen 2.0 is not available. "
                "Install with: pip install 'autogen-agentchat>=0.4.0' 'autogen-ext[openai]>=0.4.0'"
            )

        team = self._create_team_autogen2()
        chapter_prompt = self._build_chapter_prompt(chapter_number, prompt)

        # Run the team async
        try:
            result = asyncio.run(self._run_team_async(team, chapter_prompt))
            messages = self._extract_messages_from_result(result)
        except Exception as e:
            logger.error(f"AutoGen 2.0 team execution failed for chapter {chapter_number}: {e}")
            messages = []

        return messages

    async def _run_team_async(self, team: Any, message: str) -> Any:
        """Run the AutoGen 2.0 team asynchronously"""
        result = await team.run(task=message)
        return result

    def _extract_messages_from_result(self, result: Any) -> List[Dict[str, Any]]:
        """Extract messages from AutoGen 2.0 team result"""
        messages = []

        if hasattr(result, "messages"):
            for msg in result.messages:
                messages.append(self._convert_autogen2_message(msg))
        elif isinstance(result, list):
            for msg in result:
                messages.append(self._convert_autogen2_message(msg))
        else:
            logger.warning(f"Unknown result format: {type(result)}")

        return messages

    def _convert_autogen2_message(self, msg: Any) -> Dict[str, Any]:
        """Convert AutoGen 2.0 message to legacy format"""
        if hasattr(msg, "content"):
            content = msg.content
            source = getattr(msg, "source", getattr(msg, "sender", "unknown"))
            return {"content": content, "sender": source}
        elif isinstance(msg, dict):
            return msg
        else:
            return {"content": str(msg), "sender": "unknown"}

    def _find_chapter_completion_index(self, messages: List[Dict[str, Any]]) -> Optional[int]:
        """Find the index where chapter generation is complete

        Returns the index of the message that completes the chapter sequence,
        or None if the chapter is not complete.

        A chapter is considered complete when:
        1. Memory Keeper provided MEMORY_UPDATE
        2. Writer provided initial draft (SCENE_TAG)
        3. Editor provided feedback (FEEDBACK_TAG)
        4. Writer Final provided final version (SCENE_FINAL_TAG)
        """
        if not messages:
            return None

        # Track completion markers
        has_memory_update = False
        has_scene_draft = False
        has_editor_feedback = False
        has_scene_final = False

        for i, msg in enumerate(messages):
            content = msg.get("content", "")

            if AgentConstants.MEMORY_UPDATE_TAG in content:
                has_memory_update = True

            if AgentConstants.SCENE_TAG in content:
                has_scene_draft = True

            if AgentConstants.FEEDBACK_TAG in content:
                has_editor_feedback = True

            if AgentConstants.SCENE_FINAL_TAG in content:
                has_scene_final = True
                # Check if all steps are complete
                if has_memory_update and has_scene_draft and has_editor_feedback and has_scene_final:
                    return i

        return None

    def _create_team_autogen2(self) -> Any:
        """Create a new team for the agents (AutoGen 2.0)"""
        if RoundRobinGroupChat is None or MaxMessageTermination is None or AssistantAgent is None:
            raise ImportError("AutoGen 2.0 is not available")

        outline_context = "\n".join([
            f"\nChapter {ch['chapter_number']}: {ch['title']}\n{ch['prompt']}"
            for ch in sorted(self.outline, key=lambda x: x["chapter_number"])
        ])

        # Create writer_final agent (clone of writer with same config)
        writer = self.agents.get("writer")
        if writer is None:
            raise ValueError("Writer agent not found")

        model_client = getattr(writer, "model_client", None)
        if model_client is None:
            raise ValueError("Writer agent must have a model_client for AutoGen 2.0")

        writer_final = AssistantAgent(
            name="writer_final",
            model_client=model_client,
            system_message=writer.system_message,
        )

        # Build participants list, ensuring no None values
        participants = []
        for agent_name in ["user_proxy", "memory_keeper", "writer", "editor"]:
            agent = self.agents.get(agent_name)
            if agent is not None:
                participants.append(agent)

        # Add writer_final at the end
        participants.append(writer_final)

        termination = MaxMessageTermination(10)  # Limit rounds to prevent excessive continuation

        return RoundRobinGroupChat(
            participants=participants,
            termination_condition=termination,
        )

    def _save_checkpoint(self, chapter_number: int, stage: str, data: Dict[str, Any]) -> None:
        """Save checkpoint data for debugging and resume support"""
        import json
        from datetime import datetime

        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "chapter_number": chapter_number,
            "stage": stage,
            "data": data
        }

        filename = os.path.join(checkpoint_dir, f"chapter_{chapter_number:02d}_{stage}.json")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, default=str)
            logger.debug(f"Checkpoint saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _build_chapter_prompt(self, chapter_number: int, prompt: str) -> str:
        """Prepare context for chapter generation"""
        context = self._prepare_chapter_context(chapter_number, prompt)
        is_last = chapter_number >= len(self.outline)

        return f"""IMPORTANT: Wait for confirmation before proceeding.
IMPORTANT: This is Chapter {chapter_number}. Do not proceed to next chapter until explicitly instructed.
{"End the story here." if is_last else "DO NOT END THE STORY HERE."}

Current Task: Generate Chapter {chapter_number} content only.

Chapter Outline:
Title: {self.outline[chapter_number - 1]['title']}

Chapter Requirements:
{prompt}

Previous Context for Reference:
{context}

Follow this exact sequence for Chapter {chapter_number} only:
1. Memory Keeper: Context (MEMORY UPDATE)
2. Writer: Draft (CHAPTER)
3. Editor: Review (FEEDBACK)
4. Writer Final: Revision (CHAPTER FINAL)

Wait for each step to complete before proceeding."""

    def _prepare_chapter_context(self, chapter_number: int, prompt: str) -> str:
        """Prepare context for chapter generation with QMD search support"""
        if chapter_number == 1:
            return f"Initial Chapter\nRequirements:\n{prompt}"

        context_parts = [
            "Previous Chapter Summaries:",
            *[
                f"Chapter {i+1}: {summary if isinstance(summary, str) else str(summary)}"
                for i, summary in enumerate(self.chapters_memory)
            ],
        ]

        # Add QMD search context if available
        qmd_context = self._get_qmd_continuity_context(chapter_number, prompt)
        if qmd_context:
            context_parts.extend(["\nRelevant Previous Content (from search):", qmd_context])

        context_parts.extend([
            "\nCurrent Chapter Requirements:",
            str(prompt) if not isinstance(prompt, str) else prompt
        ])
        return "\n".join(context_parts)

    def _get_qmd_continuity_context(self, chapter_number: int, prompt: str) -> str:
        """Get continuity context from QMD search

        Args:
            chapter_number: Current chapter number
            prompt: Chapter requirements to extract search terms from

        Returns:
            Formatted context string or empty string if QMD not available
        """
        if not self.qmd_manager or not self.qmd_manager.is_ready():
            return ""

        try:
            # Extract key terms from prompt for search
            # Look for character names, plot points, etc.
            import re

            # Search for character references
            character_pattern = re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.MULTILINE)
            potential_names = character_pattern.findall(prompt)

            context_parts = []

            # Search for main character mentions
            for name in potential_names[:2]:  # Limit to first 2 names
                if len(name) > 3:  # Filter out short words
                    results = self.qmd_manager.search_characters(name)
                    if results:
                        formatted = self.qmd_manager.format_search_results_for_agent(
                            results, context_type=f"character '{name}'"
                        )
                        if "No character information found" not in formatted:
                            context_parts.append(formatted)

            # Search for plot-related content
            plot_keywords = [
                "plot", "event", "happens", "scene", "moment",
                "climax", "conflict", "resolution", "discovery"
            ]
            for keyword in plot_keywords[:2]:
                if keyword.lower() in prompt.lower():
                    results = self.qmd_manager.search_chapters(
                        f"{keyword} in previous chapters",
                        max_results=2
                    )
                    if results:
                        formatted = self.qmd_manager.format_search_results_for_agent(
                            results, context_type="plot event"
                        )
                        if "No plot event information found" not in formatted:
                            context_parts.append(formatted)
                            break  # Only add one plot context

            if context_parts:
                return "\n\n".join(context_parts)

        except Exception as e:
            logger.debug(f"QMD context search error: {e}")

        return ""

    def _save_intermediate_drafts(
        self, chapter_number: int, messages: List[Dict[str, Any]]
    ) -> None:
        """
        Extract and save intermediate drafts from conversation messages.

        Saves drafts at different stages:
        - Initial writer drafts (scene tags)
        - Editor feedback versions
        - Final polished versions

        Args:
            chapter_number: The chapter number being processed
            messages: List of conversation messages
        """
        from pathlib import Path

        logger.debug(f"Extracting intermediate drafts for chapter {chapter_number}")

        # Create drafts directory
        drafts_dir = Path(self.output_dir) / FileConstants.DRAFTS_SUBDIR
        drafts_dir.mkdir(parents=True, exist_ok=True)

        draft_count = 0
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue

            # Extract content from various tags
            draft_content = None
            draft_type = None

            # Look for scene content (initial draft)
            if AgentConstants.SCENE_TAG in content:
                draft_content = extract_content_between_tags(
                    content,
                    AgentConstants.SCENE_TAG,
                    AgentConstants.CHAPTER_END_TAG
                )
                draft_type = "writer_draft"
                sender = get_sender_from_message(msg)
                if "editor" in sender.lower():
                    draft_type = "editor_revision"

            # Look for final scene content
            elif AgentConstants.SCENE_FINAL_TAG in content:
                draft_content = extract_content_between_tags(
                    content,
                    AgentConstants.SCENE_FINAL_TAG,
                    AgentConstants.CHAPTER_END_TAG
                )
                draft_type = "final_draft"

            # Look for content in chapter tags
            elif AgentConstants.CHAPTER_START_TAG in content:
                draft_content = extract_content_between_tags(
                    content,
                    AgentConstants.CHAPTER_START_TAG,
                    AgentConstants.CHAPTER_END_TAG
                )
                draft_type = "tagged_content"

            # Save draft if content found and substantial
            if draft_content and len(draft_content.strip()) > 100:
                draft_count += 1
                draft_filename = f"chapter_{chapter_number:02d}_draft_{draft_count:02d}_{draft_type}.md"
                draft_path = drafts_dir / draft_filename

                try:
                    with open(draft_path, 'w', encoding=FileConstants.DEFAULT_ENCODING) as f:
                        f.write(f"<!-- Draft {draft_count} - {draft_type} -->")
                        f.write(f"<!-- Message index: {msg_idx} -->")
                        f.write(f"<!-- Sender: {get_sender_from_message(msg)} -->\n\n")
                        f.write(draft_content.strip())

                    word_count = count_words(draft_content)
                    logger.info(
                        f"Saved {draft_type} for chapter {chapter_number} "
                        f"({word_count} words) to {draft_filename}"
                    )
                except Exception as e:
                    logger.error(f"Failed to save draft {draft_count} for chapter {chapter_number}: {e}")

        if draft_count > 0:
            logger.info(f"Saved {draft_count} intermediate drafts for chapter {chapter_number}")
        else:
            logger.warning(f"No intermediate drafts found for chapter {chapter_number}")


    def _save_conversation_log(self, chapter_number: int, messages: List[Dict[str, Any]]) -> None:
        """Save conversation log for debugging"""
        import json

        log_dir = os.path.join(self.output_dir, "conversation_logs")
        os.makedirs(log_dir, exist_ok=True)

        filename = os.path.join(log_dir, f"chapter_{chapter_number:02d}_conversation.json")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, default=str)
            logger.debug(f"Conversation log saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save conversation log: {e}")

    def _process_chapter_results(
        self, chapter_number: int, messages: List[Dict[str, Any]]
    ) -> None:
        """Process and save chapter results, updating memory"""
        logger.debug(f"Processing chapter {chapter_number} results")

        # Save conversation log for debugging
        self._save_conversation_log(chapter_number, messages)

        # Extract and save intermediate drafts
        self._save_intermediate_drafts(chapter_number, messages)

        # Extract Memory Keeper's summary
        memory_update = None
        for msg in reversed(messages):
            sender = get_sender_from_message(msg)
            content = msg.get("content", "")

            if sender == "memory_keeper" and AgentConstants.MEMORY_UPDATE_TAG in content:
                parts = content.split(AgentConstants.MEMORY_UPDATE_TAG)
                if len(parts) > 1:
                    memory_update = parts[1].strip()
                break

        # Add to memory - ensure it's always a string
        if memory_update:
            # Ensure memory_update is a string, not a list
            if isinstance(memory_update, list):
                memory_update = " ".join(str(item) for item in memory_update)
            self.chapters_memory.append(str(memory_update))
            logger.debug(f"Added memory update for chapter {chapter_number}: {memory_update[:100]}...")
        else:
            # Create basic summary from chapter content
            content = self._extract_final_scene(messages)
            if content:
                basic_summary = f"Chapter {chapter_number} Summary: {content[:200]}..."
                self.chapters_memory.append(basic_summary)
                logger.debug(f"Added basic summary for chapter {chapter_number}")

        # Extract and save chapter content
        self._save_chapter(chapter_number, messages)

    @retry_with_backoff(max_retries=2, base_delay=2.0)
    def _save_chapter(self, chapter_number: int, messages) -> None:
        """Save chapter content to file and index in QMD"""
        logger.info(f"Saving Chapter {chapter_number}")

        # DEBUG: Log what we received
        logger.debug(f"_save_chapter called with chapter_number={chapter_number}, messages type={type(messages)}")

        if isinstance(messages, list):
            chapter_content = self._extract_final_scene(messages)
            logger.debug(f"Extracted content from messages: {len(chapter_content) if chapter_content else 0} chars")
        else:
            chapter_content = messages  # Assume it's already content string
            logger.debug(f"Using provided content string: {len(chapter_content) if chapter_content else 0} chars")

        if not chapter_content:
            logger.error(f"No content extracted for Chapter {chapter_number}")
            raise FileOperationError(
                f"No content found for Chapter {chapter_number}"
            )

        chapter_content = clean_chapter_content(chapter_content)
        logger.debug(f"Content after cleaning: {len(chapter_content)} chars")

        # Save checkpoint before file operations
        self._save_checkpoint(chapter_number, "content_extracted", {
            "word_count": len(chapter_content.split()),
            "content_preview": chapter_content[:500]
        })

        # Validate content length
        word_count = count_words(chapter_content)
        logger.debug(f"Word count: {word_count}, minimum required: {ChapterConstants.MIN_WORD_COUNT}")
        if word_count < ChapterConstants.MIN_WORD_COUNT:
            suggestion = (
                f"Chapter {chapter_number} is too short ({word_count} words, minimum {ChapterConstants.MIN_WORD_COUNT}). "
                f"Suggestions:\n"
                f"1. Check LLM output - it may have timed out or produced incomplete content\n"
                f"2. Increase timeout in constants.py (currently {ConfigConstants.DEFAULT_TIMEOUT}s)\n"
                f"3. Reduce BOOK_MIN_WORDS in .env if your LLM cannot generate this much content\n"
                f"4. Check conversation logs in book_output/conversation_logs/\n"
                f"5. Try using a more capable model for creative tasks"
            )
            raise ChapterTooShortError(
                suggestion,
                chapter_number=chapter_number,
                word_count=word_count,
                min_words=ChapterConstants.MIN_WORD_COUNT
            )

        filename = os.path.join(
            self.output_dir, format_chapter_filename(chapter_number)
        )

        # DEBUG: Log file path
        logger.debug(f"Output directory: {self.output_dir}")
        logger.debug(f"Target filename: {filename}")
        logger.debug(f"Absolute path: {os.path.abspath(filename)}")

        # Create backup if file exists
        if os.path.exists(filename):
            backup_filename = f"{filename}{FileConstants.BACKUP_EXTENSION}"
            shutil.copy2(filename, backup_filename)
            logger.debug(f"Created backup: {backup_filename}")

        try:
            logger.debug(f"Attempting to write file: {filename}")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Chapter {chapter_number}\n\n{chapter_content}")
            logger.debug("File written successfully")

            # Verify file
            logger.debug("Verifying file was created...")
            with open(filename, "r", encoding="utf-8") as f:
                saved_content = f.read()
                if len(saved_content.strip()) == 0:
                    raise FileOperationError(
                        f"File {filename} is empty",
                        filepath=filename
                    )

            logger.info(f"✓ Saved to: {filename}")

            # Index chapter in QMD for search
            self._index_chapter_in_qmd(chapter_number, chapter_content)

        except IOError as e:
            logger.error(f"IOError while saving chapter: {e}")
            raise FileOperationError(
                f"Error saving chapter: {e}",
                filepath=filename,
                operation="save"
            ) from e

    def _index_chapter_in_qmd(self, chapter_number: int, content: str) -> None:
        """Index chapter in QMD for search capabilities

        Args:
            chapter_number: Chapter number
            content: Chapter content to index
        """
        if not self.qmd_manager or not self.qmd_manager.is_ready():
            return

        try:
            success = self.qmd_manager.index_chapter(chapter_number, content)
            if success:
                logger.debug(f"Chapter {chapter_number} indexed in QMD")
            else:
                logger.warning(f"Failed to index Chapter {chapter_number} in QMD")
        except Exception as e:
            # Don't let QMD indexing failures stop the book generation
            logger.warning(f"QMD indexing error for Chapter {chapter_number}: {e}")

    def _extract_final_scene(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract chapter content with improved content detection"""
        logger.debug(f"Extracting final scene from {len(messages)} messages")

        for msg in reversed(messages):
            content = msg.get("content", "")
            sender = get_sender_from_message(msg)

            logger.debug(f"Checking message from {sender}: {len(content)} chars")

            if sender in ["writer", "writer_final"]:
                # Handle complete scene content
                if AgentConstants.SCENE_FINAL_TAG in content:
                    parts = content.split(AgentConstants.SCENE_FINAL_TAG)
                    if len(parts) > 1 and parts[1].strip():
                        logger.debug(f"Found SCENE_FINAL_TAG content: {len(parts[1])} chars")
                        return parts[1].strip()

                # Fallback to scene content
                if AgentConstants.SCENE_TAG in content:
                    parts = content.split(AgentConstants.SCENE_TAG)
                    if len(parts) > 1 and parts[1].strip():
                        logger.debug(f"Found SCENE_TAG content: {len(parts[1])} chars")
                        return parts[1].strip()

                # Handle raw content - look for substantial text that looks like a chapter
                content_stripped = content.strip()
                if len(content_stripped) > 500:  # Lowered threshold but still substantial
                    # Check if it looks like narrative content (has sentences, paragraphs)
                    if "." in content_stripped and len(content_stripped.split()) > 100:
                        logger.debug(f"Found raw content: {len(content_stripped)} chars, {len(content_stripped.split())} words")
                        return content_stripped

        logger.warning("No chapter content found in any message")
        # Debug: log all writer messages to see what we got
        for msg in messages:
            sender = get_sender_from_message(msg)
            if sender in ["writer", "writer_final"]:
                content = msg.get("content", "")
                logger.debug(f"Writer message preview: {content[:200]}...")

        return None

    def _handle_chapter_generation_failure(
        self, chapter_number: int, prompt: str
    ) -> None:
        """Handle failed chapter generation with simplified retry.

        This method is called when the standard multi-agent chapter generation
        process fails. It creates a minimal agent group (user_proxy + writer)
        and attempts to generate the chapter with a simplified, more direct prompt.

        Args:
            chapter_number: The chapter number that failed to generate
            prompt: The original chapter requirements/prompt

        Raises:
            ChapterError: If the retry attempt also fails

        Example:
            If chapter 5 fails during normal generation, this will:
            1. Create a simplified retry group chat
            2. Send an emergency prompt to the writer agent
            3. Attempt to extract and save the generated content
            4. Raise ChapterError if retry also fails
        """
        logger.warning(f"Attempting simplified retry for Chapter {chapter_number}")

        try:
            messages = self._retry_generation_autogen2(chapter_number, prompt)

            # Log what we got for debugging
            logger.debug(f"Retry produced {len(messages)} messages")
            for i, msg in enumerate(messages[-3:]):
                sender = get_sender_from_message(msg)
                content = msg.get("content", "")
                logger.debug(f"Retry message {i} from {sender}: {len(content)} chars")

            # Save the retry results
            self._process_chapter_results(chapter_number, messages)

        except Exception as e:
            logger.error(f"Error in retry attempt for Chapter {chapter_number}: {e}")
            raise ChapterError(
                "Unable to generate chapter content after retry",
                chapter_number=chapter_number
            ) from e

    def _retry_generation_autogen2(self, chapter_number: int, prompt: str) -> List[Dict[str, Any]]:
        """Retry generation using AutoGen 2.0"""
        if not AUTOGEN_2_AVAILABLE:
            raise ImportError("AutoGen 2.0 is not available")

        # Build participants list for retry
        participants = []
        for agent_name in ["user_proxy", "writer"]:
            agent = self.agents.get(agent_name)
            if agent is not None:
                participants.append(agent)

        # Add memory_keeper if available
        if "memory_keeper" in self.agents:
            agent = self.agents.get("memory_keeper")
            if agent is not None:
                participants.insert(1, agent)

        termination = MaxMessageTermination(GroupChatConstants.REPLY_MAX_ROUNDS)

        team = RoundRobinGroupChat(
            participants=participants,
            termination_condition=termination,
        )

        retry_prompt = f"""EMERGENCY CHAPTER GENERATION for Chapter {chapter_number}.

Previous attempt failed. Generate this chapter NOW.

Chapter Requirements:
{prompt}

INSTRUCTIONS:
1. Write the complete chapter content
2. Make it at least 3000 words
3. Start with: "Chapter {chapter_number}: [Title]"
4. End with: "END OF CHAPTER {chapter_number}"
5. Just write the story - no outlines, no planning, no meta-commentary

WRITE THE CHAPTER NOW."""

        try:
            result = asyncio.run(self._run_team_async(team, retry_prompt))
            messages = self._extract_messages_from_result(result)
        except Exception as e:
            logger.error(f"AutoGen 2.0 retry failed: {e}")
            messages = []

        return messages

    def get_book_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the generated book.

        Scans the output directory and analyzes all generated chapters,
        calculating total word count, chapter count, and per-chapter statistics.

        Returns:
            Dictionary containing:
            - total_chapters: Number of chapters found
            - total_words: Total word count across all chapters
            - chapters: List of chapter statistics including:
              - chapter: Chapter number
              - filename: Chapter filename
              - words: Word count for this chapter

        Example:
            >>> stats = generator.get_book_stats()
            >>> print(f"Book has {stats['total_chapters']} chapters")
            >>> print(f"Total words: {stats['total_words']:,}")

        Note:
            Skips files that cannot be read and logs warnings for them.
            Only processes files matching the chapter filename pattern.
        """
        stats = {
            "total_chapters": 0,
            "total_words": 0,
            "chapters": []
        }

        for filename in os.listdir(self.output_dir):
            if filename.startswith(FileConstants.CHAPTER_PREFIX):
                filepath = os.path.join(self.output_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        word_count = count_words(content)
                        match = re.search(r"\d+", filename)
                        chapter_num = int(match.group()) if match else 0

                        stats["total_chapters"] += 1
                        stats["total_words"] += word_count
                        stats["chapters"].append({
                            "chapter": chapter_num,
                            "filename": filename,
                            "words": word_count
                        })
                except Exception as e:
                    logger.warning(f"Error reading {filename}: {e}")

        return stats
