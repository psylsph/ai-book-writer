"""Main class for generating books using AutoGen with improved error handling"""
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

import autogen

from constants import (
    AgentConstants,
    ChapterConstants,
    FileConstants,
    GroupChatConstants,
    RegexPatterns,
)
from exceptions import (
    ChapterError,
    ChapterIncompleteError,
    ChapterTooShortError,
    FileOperationError,
    RetryExhaustedError,
)
from models import ChapterContent
from utils import (
    check_sequence_completion,
    clean_chapter_content,
    count_words,
    format_chapter_filename,
    get_logger,
    get_sender_from_message,
    retry_with_backoff,
    setup_logging,
    validate_chapter_length,
)


logger = get_logger("book_generator")


class BookGenerator:
    """Generates book chapters using multi-agent collaboration with validation"""

    def __init__(
        self,
        agents: Dict[str, autogen.ConversableAgent],
        agent_config: Dict[str, Any],
        outline: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ):
        """Initialize with outline to maintain chapter count context"""
        self.agents = agents
        self.agent_config = agent_config
        self.output_dir = output_dir or FileConstants.OUTPUT_DIR
        self.chapters_memory: List[str] = []
        self.max_iterations = 3
        self.outline = outline

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"BookGenerator initialized with output dir: {self.output_dir}")

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

        try:
            groupchat = self._create_group_chat()
            manager = autogen.GroupChatManager(
                groupchat=groupchat, llm_config=self.agent_config
            )

            chapter_prompt = self._build_chapter_prompt(chapter_number, prompt)

            # Start generation
            self.agents["user_proxy"].initiate_chat(manager, message=chapter_prompt)

            # Verify chapter completion
            if not self._verify_chapter_complete(groupchat.messages, chapter_number):
                raise ChapterIncompleteError(
                    f"Chapter {chapter_number} generation incomplete",
                    chapter_number=chapter_number,
                    missing_steps=self._get_missing_steps(groupchat.messages)
                )

            self._process_chapter_results(chapter_number, groupchat.messages)

            chapter_file = os.path.join(
                self.output_dir, format_chapter_filename(chapter_number)
            )
            if not os.path.exists(chapter_file):
                raise FileOperationError(
                    f"Chapter {chapter_number} file not created",
                    filepath=chapter_file,
                    operation="create"
                )

            completion_msg = f"Chapter {chapter_number} is complete. Proceed with next chapter."
            self.agents["user_proxy"].send(completion_msg, manager)

        except RetryExhaustedError:
            logger.error(f"All retry attempts exhausted for chapter {chapter_number}")
            raise
        except Exception as e:
            logger.error(f"Error in chapter {chapter_number}: {e}")
            self._save_checkpoint(chapter_number, "error", {"error": str(e), "messages": messages if 'messages' in locals() else []})
            self._handle_chapter_generation_failure(chapter_number, prompt)

    def _create_group_chat(self) -> autogen.GroupChat:
        """Create a new group chat for the agents"""
        outline_context = "\n".join([
            f"\nChapter {ch['chapter_number']}: {ch['title']}\n{ch['prompt']}"
            for ch in sorted(self.outline, key=lambda x: x["chapter_number"])
        ])

        messages = [{
            "role": "system",
            "content": f"Complete Book Outline:\n{outline_context}"
        }]

        writer_final = autogen.AssistantAgent(
            name="writer_final",
            system_message=self.agents["writer"].system_message,
            llm_config=self.agent_config
        )

        return autogen.GroupChat(
            agents=[
                self.agents["user_proxy"],
                self.agents["memory_keeper"],
                self.agents["writer"],
                self.agents["editor"],
                writer_final
            ],
            messages=messages,
            max_round=GroupChatConstants.CHAPTER_MAX_ROUNDS,
            speaker_selection_method=GroupChatConstants.SPEAKER_SELECTION
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
        """Prepare context for chapter generation"""
        if chapter_number == 1:
            return f"Initial Chapter\nRequirements:\n{prompt}"

        context_parts = [
            "Previous Chapter Summaries:",
            *[
                f"Chapter {i+1}: {summary}"
                for i, summary in enumerate(self.chapters_memory)
            ],
            "\nCurrent Chapter Requirements:",
            prompt
        ]
        return "\n".join(context_parts)

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

    def _verify_chapter_complete(
        self, messages: List[Dict[str, Any]], chapter_number: int
    ) -> bool:
        """Verify chapter completion by analyzing entire conversation context"""
        logger.debug(f"Verifying chapter {chapter_number} completion")

        sequence_complete = check_sequence_completion(messages)
        chapter_content = None
        current_chapter = None

        # Find chapter number and content
        for msg in messages:
            content = msg.get("content", "")
            sender = get_sender_from_message(msg)

            # Track chapter number
            if not current_chapter:
                match = RegexPatterns.CHAPTER_NUMBER.search(content)
                if match:
                    current_chapter = int(match.group(1))

            # Track final scene content
            if sender in ["writer", "writer_final"]:
                if AgentConstants.SCENE_FINAL_TAG in content:
                    parts = content.split(AgentConstants.SCENE_FINAL_TAG)
                    if len(parts) > 1:
                        chapter_content = parts[1].strip()
                elif AgentConstants.SCENE_TAG in content:
                    parts = content.split(AgentConstants.SCENE_TAG)
                    if len(parts) > 1 and not chapter_content:
                        chapter_content = parts[1].strip()

        # Verify all steps completed and content exists
        if all(sequence_complete.values()) and current_chapter == chapter_number and chapter_content:
            self._save_chapter(chapter_number, chapter_content)
            return True

        logger.warning(
            f"Chapter {chapter_number} incomplete. "
            f"Steps: {sequence_complete}, has_content: {bool(chapter_content)}"
        )
        return False

    def _get_missing_steps(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Get list of missing steps in chapter generation"""
        sequence_complete = check_sequence_completion(messages)
        return [step for step, complete in sequence_complete.items() if not complete]

    def _process_chapter_results(
        self, chapter_number: int, messages: List[Dict[str, Any]]
    ) -> None:
        """Process and save chapter results, updating memory"""
        logger.debug(f"Processing chapter {chapter_number} results")
        
        # Save conversation log for debugging
        self._save_conversation_log(chapter_number, messages)

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

        # Add to memory
        if memory_update:
            self.chapters_memory.append(memory_update)
        else:
            # Create basic summary from chapter content
            content = self._extract_final_scene(messages)
            if content:
                basic_summary = f"Chapter {chapter_number} Summary: {content[:200]}..."
                self.chapters_memory.append(basic_summary)

        # Extract and save chapter content
        self._save_chapter(chapter_number, messages)

    @retry_with_backoff(max_retries=2, base_delay=2.0)
    def _save_chapter(self, chapter_number: int, messages) -> None:
        """Save chapter content to file"""
        logger.info(f"Saving Chapter {chapter_number}")

        if isinstance(messages, list):
            chapter_content = self._extract_final_scene(messages)
        else:
            chapter_content = messages  # Assume it's already content string

        if not chapter_content:
            raise FileOperationError(
                f"No content found for Chapter {chapter_number}"
            )

        chapter_content = clean_chapter_content(chapter_content)
        
        # Save checkpoint before file operations
        self._save_checkpoint(chapter_number, "content_extracted", {
            "word_count": len(chapter_content.split()),
            "content_preview": chapter_content[:500]
        })

        # Validate content length
        word_count = count_words(chapter_content)
        if word_count < ChapterConstants.MIN_WORD_COUNT:
            raise ChapterTooShortError(
                f"Chapter {chapter_number} too short",
                chapter_number=chapter_number,
                word_count=word_count,
                min_words=ChapterConstants.MIN_WORD_COUNT
            )

        filename = os.path.join(
            self.output_dir, format_chapter_filename(chapter_number)
        )

        # Create backup if file exists
        if os.path.exists(filename):
            backup_filename = f"{filename}{FileConstants.BACKUP_EXTENSION}"
            shutil.copy2(filename, backup_filename)
            logger.debug(f"Created backup: {backup_filename}")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Chapter {chapter_number}\n\n{chapter_content}")

            # Verify file
            with open(filename, "r", encoding="utf-8") as f:
                saved_content = f.read()
                if len(saved_content.strip()) == 0:
                    raise FileOperationError(
                        f"File {filename} is empty",
                        filepath=filename
                    )

            logger.info(f"✓ Saved to: {filename}")

        except IOError as e:
            raise FileOperationError(
                f"Error saving chapter: {e}",
                filepath=filename,
                operation="save"
            ) from e

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
        """Handle failed chapter generation with simplified retry"""
        logger.warning(f"Attempting simplified retry for Chapter {chapter_number}")

        try:
            # Create a new group chat with just essential agents
            # Note: story_planner may not be in self.agents, use writer directly
            available_agents = [
                self.agents["user_proxy"],
                self.agents["writer"]
            ]
            
            # Add memory_keeper if available
            if "memory_keeper" in self.agents:
                available_agents.insert(1, self.agents["memory_keeper"])

            retry_groupchat = autogen.GroupChat(
                agents=available_agents,
                messages=[],
                max_round=GroupChatConstants.REPLY_MAX_ROUNDS,
                speaker_selection_method=GroupChatConstants.SPEAKER_SELECTION
            )

            manager = autogen.GroupChatManager(
                groupchat=retry_groupchat,
                llm_config=self.agent_config
            )

            retry_prompt = f"""EMERGENCY CHAPTER GENERATION for Chapter {chapter_number}.

Previous attempt failed. Generate this chapter NOW.

Chapter Requirements:
{prompt}

INSTRUCTIONS:
1. Write the complete chapter content
2. Make it at least 5000 words
3. Start with: "Chapter {chapter_number}: [Title]"
4. End with: "END OF CHAPTER {chapter_number}"
5. Just write the story - no outlines, no planning, no meta-commentary

WRITE THE CHAPTER NOW."""

            self.agents["user_proxy"].initiate_chat(
                manager, message=retry_prompt
            )

            # Log what we got for debugging
            logger.debug(f"Retry produced {len(retry_groupchat.messages)} messages")
            for i, msg in enumerate(retry_groupchat.messages[-3:]):
                sender = get_sender_from_message(msg)
                content = msg.get("content", "")
                logger.debug(f"Retry message {i} from {sender}: {len(content)} chars")

            # Save the retry results
            self._process_chapter_results(chapter_number, retry_groupchat.messages)

        except Exception as e:
            logger.error(f"Error in retry attempt for Chapter {chapter_number}: {e}")
            raise ChapterError(
                f"Unable to generate chapter content after retry",
                chapter_number=chapter_number
            ) from e

    def get_book_stats(self) -> Dict[str, Any]:
        """Get statistics for the generated book"""
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
