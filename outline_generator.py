"""Generate book outlines using AutoGen agents with improved error handling and validation"""
from typing import Any, Dict, List, Optional

import autogen

from constants import (
    AgentConstants,
    ChapterConstants,
    FileConstants,
    GroupChatConstants,
    OutlineConstants,
    RegexPatterns,
)
from exceptions import ConfigurationError, ParseError, ValidationError
from models import Chapter, Outline
from utils import (
    check_sequence_completion,
    get_logger,
    get_sender_from_message,
    retry_with_backoff,
    verify_chapter_sequence,
)


logger = get_logger("outline_generator")


class OutlineGenerator:
    """Generates comprehensive book outlines using multi-agent collaboration"""

    def __init__(
        self,
        agents: Dict[str, autogen.ConversableAgent],
        agent_config: Dict[str, Any],
    ):
        self.agents = agents
        self.agent_config = agent_config
        self.num_chapters = OutlineConstants.DEFAULT_NUM_CHAPTERS
        logger.info("OutlineGenerator initialized")

    def generate_outline(
        self, initial_prompt: str, num_chapters: int = 25
    ) -> List[Dict[str, Any]]:
        """Generate a book outline based on initial prompt with validation"""
        logger.info(f"Generating outline with {num_chapters} chapters")
        
        if num_chapters < 1:
            raise ConfigurationError("Number of chapters must be at least 1")
        if num_chapters > 100:
            logger.warning(f"Large chapter count requested: {num_chapters}")

        self.num_chapters = num_chapters

        try:
            groupchat = self._create_outline_groupchat()
            manager = autogen.GroupChatManager(
                groupchat=groupchat, llm_config=self.agent_config
            )

            outline_prompt = self._build_outline_prompt(initial_prompt, num_chapters)

            # Initiate the chat with retry logic
            self._initiate_chat_with_retry(manager, outline_prompt)

            # Extract and validate the outline
            chapters = self._process_outline_results(groupchat.messages, num_chapters)
            
            if not chapters:
                logger.warning("Normal processing failed, attempting emergency processing")
                chapters = self._emergency_outline_processing(groupchat.messages, num_chapters)

            # Validate final outline
            if not chapters:
                raise ParseError("Failed to extract any chapters from outline")

            # Sort and verify sequence
            chapters.sort(key=lambda x: x["chapter_number"])
            is_valid, missing = verify_chapter_sequence(chapters, num_chapters)
            
            if not is_valid:
                if missing:
                    logger.warning(f"Missing chapters: {missing}")
                raise ValidationError(
                    f"Expected {num_chapters} chapters, got {len(chapters)}",
                    field="chapter_count",
                    value=len(chapters)
                )

            logger.info(f"Successfully generated outline with {len(chapters)} chapters")
            return chapters

        except Exception as e:
            logger.error(f"Failed to generate outline: {e}")
            raise

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def _initiate_chat_with_retry(
        self, manager: autogen.GroupChatManager, prompt: str
    ) -> None:
        """Initiate chat with retry logic"""
        self.agents["user_proxy"].initiate_chat(manager, message=prompt)

    def _create_outline_groupchat(self) -> autogen.GroupChat:
        """Create a configured group chat for outline generation"""
        return autogen.GroupChat(
            agents=[
                self.agents["user_proxy"],
                self.agents["story_planner"],
                self.agents["world_builder"],
                self.agents["outline_creator"],
            ],
            messages=[],
            max_round=GroupChatConstants.OUTLINE_MAX_ROUNDS,
            speaker_selection_method=GroupChatConstants.SPEAKER_SELECTION,
        )

    def _build_outline_prompt(self, initial_prompt: str, num_chapters: int) -> str:
        """Construct the prompt for outline generation"""
        return f"""Let's create a {num_chapters}-chapter outline for a book with the following premise:

{initial_prompt}

Process:
1. Story Planner: Create a high-level story arc and major plot points
2. World Builder: Suggest key settings and world elements needed
3. Outline Creator: Generate a detailed outline with chapter titles and prompts

Requirements:
- Each chapter MUST have exactly these fields: Title, Key Events, Character Developments, Setting, Tone
- Every chapter MUST have at least {OutlineConstants.MIN_EVENTS_PER_CHAPTER} Key Events
- Total chapters in outline: {num_chapters}
- Do not combine chapters
- Do not leave any chapters undefined
- Think through every chapter carefully

Start with Chapter 1 and number chapters sequentially.

End the outline with '{AgentConstants.OUTLINE_END_TAG}'"""

    def _process_outline_results(
        self, messages: List[Dict[str, Any]], num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Extract and process the outline with strict format requirements"""
        logger.debug("Processing outline from messages")
        outline_content = self._extract_outline_content(messages)

        if not outline_content:
            logger.warning("No structured outline found")
            return []

        chapters = self._parse_chapter_sections(outline_content, num_chapters)
        
        if len(chapters) < num_chapters:
            logger.warning(
                f"Only extracted {len(chapters)} chapters out of {num_chapters} required"
            )

        return chapters

    def _extract_outline_content(self, messages: List[Dict[str, Any]]) -> str:
        """Extract outline content from messages"""
        logger.debug(f"Searching {len(messages)} messages for outline content")

        # Look for content between "OUTLINE:" and "END OF OUTLINE"
        for msg in reversed(messages):
            content = msg.get("content", "")
            if AgentConstants.OUTLINE_START_TAG in content:
                start_idx = content.find(AgentConstants.OUTLINE_START_TAG)
                end_idx = content.find(AgentConstants.OUTLINE_END_TAG)

                if start_idx != -1:
                    if end_idx != -1:
                        return content[start_idx:end_idx].strip()
                    else:
                        return content[start_idx:].strip()

        # Fallback: look for content with chapter markers
        for msg in reversed(messages):
            content = msg.get("content", "")
            if "Chapter 1:" in content or "**Chapter 1:**" in content:
                return content

        return ""

    def _parse_chapter_sections(
        self, outline_content: str, num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Parse individual chapter sections from outline content"""
        chapters = []
        chapter_sections = RegexPatterns.CHAPTER_SPLIT.split(outline_content)

        for i, section in enumerate(chapter_sections[1:], 1):  # Skip first empty section
            try:
                chapter_info = self._extract_chapter_components(section, i)
                if chapter_info:
                    chapters.append(chapter_info)
            except ValidationError as e:
                logger.warning(f"Chapter {i} validation error: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing Chapter {i}: {e}")
                continue

        return chapters

    def _extract_chapter_components(
        self, section: str, chapter_num: int
    ) -> Optional[Dict[str, Any]]:
        """Extract components from a single chapter section"""
        # Extract required components using regex
        title_match = RegexPatterns.TITLE.search(section)
        title_alt_match = RegexPatterns.CHAPTER_TITLE_ALT.search(section)
        events_match = RegexPatterns.KEY_EVENTS.search(section)
        character_match = RegexPatterns.CHARACTER_DEVELOPMENTS.search(section)
        setting_match = RegexPatterns.SETTING.search(section)
        tone_match = RegexPatterns.TONE.search(section)

        # Use alternate title if primary not found
        if not title_match and title_alt_match:
            title_match = title_alt_match

        # Verify all components exist
        if not all([title_match, events_match, character_match, setting_match, tone_match]):
            missing = []
            if not title_match:
                missing.append("Title")
            if not events_match:
                missing.append("Key Events")
            if not character_match:
                missing.append("Character Developments")
            if not setting_match:
                missing.append("Setting")
            if not tone_match:
                missing.append("Tone")
            logger.warning(f"Chapter {chapter_num} missing components: {', '.join(missing)}")
            raise ValidationError(
                f"Missing components: {', '.join(missing)}",
                field="components",
                value=missing
            )

        # Extract events and verify count
        events_text = events_match.group(1).strip() if events_match else ""
        events = RegexPatterns.BULLET_POINT.findall(events_text)
        
        if len(events) < OutlineConstants.MIN_EVENTS_PER_CHAPTER:
            raise ValidationError(
                f"Chapter {chapter_num} has fewer than {OutlineConstants.MIN_EVENTS_PER_CHAPTER} events",
                field="events",
                value=len(events)
            )

        # Build chapter info
        title = title_match.group(1).strip() if title_match else f"Chapter {chapter_num}"
        prompt = "\n".join([
            f"- Key Events: {events_text}",
            f"- Character Developments: {character_match.group(1).strip() if character_match else '[To be determined]'}",
            f"- Setting: {setting_match.group(1).strip() if setting_match else '[To be determined]'}",
            f"- Tone: {tone_match.group(1).strip() if tone_match else '[To be determined]'}",
        ])

        return {
            "chapter_number": chapter_num,
            "title": title,
            "prompt": prompt,
        }

    def _emergency_outline_processing(
        self, messages: List[Dict[str, Any]], num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Emergency processing when normal outline extraction fails"""
        logger.warning("Attempting emergency outline processing")

        chapters = []
        current_chapter: Optional[Dict[str, Any]] = None

        # Look through all messages for any chapter content
        for msg in messages:
            content = msg.get("content", "")
            lines = content.split("\n")

            for line in lines:
                # Look for chapter markers
                chapter_match = RegexPatterns.CHAPTER_NUMBER.search(line)
                if chapter_match and "Key events:" in content.lower():
                    if current_chapter and current_chapter.get("prompt"):
                        chapters.append(current_chapter)

                    chapter_num = int(chapter_match.group(1))
                    title = line.split(":")[-1].strip() if ":" in line else f"Chapter {chapter_num}"
                    current_chapter = {
                        "chapter_number": chapter_num,
                        "title": title,
                        "prompt": [],
                    }

                # Collect bullet points
                if current_chapter and line.strip().startswith("-"):
                    if isinstance(current_chapter["prompt"], list):
                        current_chapter["prompt"].append(line.strip())

            # Add the last chapter if it exists
            if current_chapter and current_chapter.get("prompt"):
                if isinstance(current_chapter["prompt"], list):
                    current_chapter["prompt"] = "\n".join(current_chapter["prompt"])
                chapters.append(current_chapter)
                current_chapter = None

        if not chapters:
            logger.error("Emergency processing failed to find any chapters")
            return []

        # Sort and dedupe
        seen_numbers = set()
        unique_chapters = []
        for ch in sorted(chapters, key=lambda x: x["chapter_number"]):
            if ch["chapter_number"] not in seen_numbers:
                seen_numbers.add(ch["chapter_number"])
                unique_chapters.append(ch)

        return self._verify_chapter_sequence(unique_chapters, num_chapters)

    def _verify_chapter_sequence(
        self, chapters: List[Dict[str, Any]], num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Verify and fix chapter numbering"""
        # Sort chapters by their current number
        chapters.sort(key=lambda x: x["chapter_number"])

        # Renumber chapters sequentially starting from 1
        for i, chapter in enumerate(chapters, 1):
            chapter["chapter_number"] = i

        # Add placeholder chapters if needed
        while len(chapters) < num_chapters:
            next_num = len(chapters) + 1
            chapters.append({
                "chapter_number": next_num,
                "title": f"Chapter {next_num}",
                "prompt": (
                    f"- Key events: Scene {next_num}a, Scene {next_num}b, Scene {next_num}c\n"
                    f"- Character developments: [To be determined]\n"
                    f"- Setting: [To be determined]\n"
                    f"- Tone: [To be determined]"
                ),
            })

        # Trim excess chapters if needed
        return chapters[:num_chapters]

    def _build_outline_from_parsed(self, chapters: List[Dict[str, Any]]) -> Outline:
        """Build validated Outline model from parsed chapters"""
        chapter_models = []
        for ch in chapters:
            try:
                chapter = Chapter(
                    chapter_number=ch["chapter_number"],
                    title=ch["title"],
                    prompt=ch["prompt"],
                )
                chapter_models.append(chapter)
            except Exception as e:
                logger.warning(f"Failed to validate chapter {ch.get('chapter_number')}: {e}")
                continue

        return Outline(
            chapters=chapter_models,
            total_chapters=self.num_chapters
        )
