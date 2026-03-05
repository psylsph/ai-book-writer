"""Generate book outlines using AutoGen agents with improved error handling and validation

Supports both legacy AutoGen 0.2 and AutoGen 2.0 (0.4+) APIs.
AutoGen 2.0 uses async team execution with termination conditions.
"""
import asyncio
import re
from typing import Any, Dict, List, Optional

# AutoGen 2.0 imports
try:
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    AUTOGEN_2_AVAILABLE = True
except ImportError:
    AUTOGEN_2_AVAILABLE = False
    RoundRobinGroupChat = None
    MaxMessageTermination = None

# Legacy AutoGen imports
try:
    import autogen as autogen_legacy
    AUTOGEN_LEGACY_AVAILABLE = True
except ImportError:
    AUTOGEN_LEGACY_AVAILABLE = False
    autogen_legacy = None

from constants import (
    AgentConstants,
    GroupChatConstants,
    OutlineConstants,
    RegexPatterns,
)
from exceptions import ConfigurationError, ParseError, ValidationError
from models import Chapter, Outline
from utils import (
    get_logger,
    retry_with_backoff,
    verify_chapter_sequence,
)

logger = get_logger("outline_generator")


class OutlineGenerator:
    """Generates comprehensive book outlines using multi-agent collaboration

    Supports both legacy AutoGen 0.2 (synchronous) and AutoGen 2.0 (async) APIs.
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        agent_config: Dict[str, Any],
        use_autogen2: bool = True,
    ):
        self.agents = agents
        self.agent_config = agent_config
        self.use_autogen2 = use_autogen2 and AUTOGEN_2_AVAILABLE
        self.num_chapters = OutlineConstants.DEFAULT_NUM_CHAPTERS
        logger.info(f"OutlineGenerator initialized (AutoGen 2.0: {self.use_autogen2})")

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
            if self.use_autogen2:
                chapters = self._generate_outline_autogen2(initial_prompt, num_chapters)
            else:
                chapters = self._generate_outline_legacy(initial_prompt, num_chapters)

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

    def _generate_outline_legacy(
        self, initial_prompt: str, num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Generate outline using legacy AutoGen (synchronous)"""
        if not AUTOGEN_LEGACY_AVAILABLE or autogen_legacy is None:
            raise ImportError("Legacy AutoGen not available. Install with: pip install pyautogen")

        groupchat = self._create_outline_groupchat_legacy()
        manager = autogen_legacy.GroupChatManager(
            groupchat=groupchat, llm_config=self.agent_config
        )

        outline_prompt = self._build_outline_prompt(initial_prompt, num_chapters)

        # Initiate the chat with retry logic
        self._initiate_chat_with_retry_legacy(manager, outline_prompt)

        # Extract and validate the outline
        chapters = self._process_outline_results(groupchat.messages, num_chapters)

        if not chapters:
            logger.warning("Normal processing failed, attempting emergency processing")
            chapters = self._emergency_outline_processing(groupchat.messages, num_chapters)

        return chapters

    def _generate_outline_autogen2(
        self, initial_prompt: str, num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Generate outline using AutoGen 2.0 (async)"""
        if not AUTOGEN_2_AVAILABLE:
            raise ImportError(
                "AutoGen 2.0 is not available. "
                "Install with: pip install 'autogen-agentchat>=0.4.0' 'autogen-ext[openai]>=0.4.0'"
            )

        team = self._create_outline_team_autogen2()

        outline_prompt = self._build_outline_prompt(initial_prompt, num_chapters)

        # Run the team async
        try:
            result = asyncio.run(self._run_team_async(team, outline_prompt))
            messages = self._extract_messages_from_result(result)
        except Exception as e:
            logger.error(f"AutoGen 2.0 team execution failed: {e}")
            # Fallback to emergency processing
            messages = []

        # Process results
        if messages:
            chapters = self._process_outline_results(messages, num_chapters)
        else:
            chapters = []

        if not chapters:
            logger.warning("AutoGen 2.0 processing failed, attempting emergency processing")
            chapters = self._emergency_outline_processing(messages, num_chapters)

        return chapters

    async def _run_team_async(self, team: Any, message: str) -> Any:
        """Run the AutoGen 2.0 team asynchronously"""
        # AutoGen 2.0 teams are run via run() method
        # Run the team with the message
        result = await team.run(task=message)
        return result

    def _extract_messages_from_result(self, result: Any) -> List[Dict[str, Any]]:
        """Extract messages from AutoGen 2.0 team result"""
        # AutoGen 2.0 results have a different structure
        messages = []

        if hasattr(result, "messages"):
            # Result has messages attribute
            for msg in result.messages:
                messages.append(self._convert_autogen2_message(msg))
        elif isinstance(result, list):
            # Result is a list of messages
            for msg in result:
                messages.append(self._convert_autogen2_message(msg))
        else:
            # Unknown format, try to extract content
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

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def _initiate_chat_with_retry_legacy(
        self, manager: Any, prompt: str
    ) -> None:
        """Initiate chat with retry logic (legacy)"""
        self.agents["user_proxy"].initiate_chat(manager, message=prompt)

    def _create_outline_groupchat_legacy(self) -> Any:
        """Create a configured group chat for outline generation (legacy)"""
        if autogen_legacy is None:
            raise ImportError("Legacy AutoGen not available")

        return autogen_legacy.GroupChat(
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

    def _create_outline_team_autogen2(self) -> Any:
        """Create a configured team for outline generation (AutoGen 2.0)"""
        if RoundRobinGroupChat is None or MaxMessageTermination is None:
            raise ImportError("AutoGen 2.0 is not available")

        participants = [
            self.agents.get("user_proxy"),
            self.agents.get("story_planner"),
            self.agents.get("world_builder"),
            self.agents.get("outline_creator"),
        ]

        # Filter out None values
        participants = [p for p in participants if p is not None]

        termination = MaxMessageTermination(GroupChatConstants.OUTLINE_MAX_ROUNDS)

        return RoundRobinGroupChat(
            participants=participants,
            termination_condition=termination,
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

        # Debug: Log all message senders to understand the conversation flow
        sender_counts = {}
        for msg in messages:
            sender = msg.get("name", msg.get("role", msg.get("sender", "unknown")))
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        logger.debug(f"Message senders: {sender_counts}")

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
            # Also look for "End of Chapter X" pattern which indicates narrative content
            if "End of Chapter" in content:
                logger.warning("Detected narrative content instead of outline format")
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
                else:
                    logger.debug(f"Chapter {i}: No extractable components found")
            except ValidationError as e:
                logger.warning(f"Chapter {i} validation error: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing Chapter {i}: {e}")
                continue

        logger.info(f"Parsed {len(chapters)} chapters from outline content")

        return chapters

    def _extract_chapter_components(
        self, section: str, chapter_num: int
    ) -> Optional[Dict[str, Any]]:
        """Extract components from a single chapter section with lenient parsing"""
        logger.debug(f"Extracting components for chapter {chapter_num}, section length: {len(section)}")

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
            logger.debug(f"Chapter {chapter_num}: Using alternate title pattern")

        # Log what we found
        found_components = []
        if title_match:
            found_components.append("Title")
        if events_match:
            found_components.append("Key Events")
        if character_match:
            found_components.append("Character Developments")
        if setting_match:
            found_components.append("Setting")
        if tone_match:
            found_components.append("Tone")

        logger.debug(f"Chapter {chapter_num} found components: {found_components}")

        # If we have at least a title and some content, create a lenient chapter
        if title_match or events_match:
            # Extract events and verify count
            events_text = events_match.group(1).strip() if events_match else "Develop the story forward"
            _ = RegexPatterns.BULLET_POINT.findall(events_text)  # Validate bullet format

            # Build chapter info with fallbacks for missing components
            title = title_match.group(1).strip() if title_match else f"Chapter {chapter_num}"

            # Build prompt with whatever we have
            prompt_parts = [f"- Key Events: {events_text}"]

            if character_match:
                prompt_parts.append(f"- Character Developments: {character_match.group(1).strip()}")
            else:
                prompt_parts.append("- Character Developments: Continue character arcs from previous chapters")

            if setting_match:
                prompt_parts.append(f"- Setting: {setting_match.group(1).strip()}")
            else:
                prompt_parts.append("- Setting: Maintain consistent world setting")

            if tone_match:
                prompt_parts.append(f"- Tone: {tone_match.group(1).strip()}")
            else:
                prompt_parts.append("- Tone: Match the established narrative tone")

            logger.info(f"Chapter {chapter_num}: Created with {len(found_components)} components (lenient mode)")

            return {
                "chapter_number": chapter_num,
                "title": title,
                "prompt": "\n".join(prompt_parts),
            }

        # If we have nothing, report what was missing
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

        # Return None to skip this chapter - emergency processing will handle it
        return None

    def _emergency_outline_processing(
        self, messages: List[Dict[str, Any]], num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Emergency processing when normal outline extraction fails"""
        logger.warning("Attempting emergency outline processing")

        # Log last few messages for debugging
        for msg in messages[-3:]:  # Last 3 messages
            sender = msg.get("name", msg.get("role", msg.get("sender", "unknown")))
            content_preview = msg.get("content", "")[:500]
            logger.debug(f"Emergency processing - {sender}: {content_preview}...")

        chapters = []
        current_chapter: Optional[Dict[str, Any]] = None

        # Look through all messages for any chapter content
        for msg in messages:
            content = msg.get("content", "")

            # Check for "End of Chapter X" pattern (indicates narrative content)
            end_chapter_matches = list(re.finditer(r'\*\*End of Chapter (\d+)\*\*', content, re.IGNORECASE))
            if end_chapter_matches:
                logger.info(f"Detected {len(end_chapter_matches)} 'End of Chapter' markers in narrative content")
                for match in end_chapter_matches:
                    chapter_num = int(match.group(1))
                    # Extract content before this marker as chapter summary
                    start_pos = max(0, content.rfind("***", 0, match.start()))
                    if start_pos == 0:
                        # Try finding start of chapter
                        chapter_start = content.rfind(f"Chapter {chapter_num}", 0, match.start())
                        if chapter_start > 0:
                            start_pos = chapter_start

                    chapter_content = content[start_pos:match.start()].strip()[:200]
                    chapters.append({
                        "chapter_number": chapter_num,
                        "title": f"Chapter {chapter_num}",
                        "prompt": (
                            f"- Key events: Continue the story from previous developments\n"
                            f"- Previous content summary: {chapter_content}...\n"
                            f"- Character developments: Build on established arcs\n"
                            f"- Setting: Maintain narrative world\n"
                            f"- Tone: Match the ongoing narrative style"
                        ),
                    })

                if chapters:
                    logger.info(f"Extracted {len(chapters)} chapters from narrative markers")
                    break  # We found chapters in this message
                continue

            # Traditional parsing for outline format
            lines = content.split("\n")
            for line in lines:
                # Look for chapter markers (more flexible matching)
                chapter_match = RegexPatterns.CHAPTER_NUMBER.search(line)
                # Accept chapter markers with or without "Key events" in same content
                if chapter_match:
                    if current_chapter and current_chapter.get("prompt"):
                        chapters.append(current_chapter)

                    chapter_num = int(chapter_match.group(1))
                    title = line.split(":")[-1].strip() if ":" in line else f"Chapter {chapter_num}"
                    current_chapter = {
                        "chapter_number": chapter_num,
                        "title": title,
                        "prompt": [],
                    }

                # Collect bullet points or any content after chapter marker
                if current_chapter:
                    line_stripped = line.strip()
                    # Accept bullet points or lines with content
                    if line_stripped.startswith("-") or line_stripped.startswith("*"):
                        if isinstance(current_chapter["prompt"], list):
                            current_chapter["prompt"].append(line_stripped)
                    # Also capture any non-empty line that looks like content
                    elif len(line_stripped) > 10 and not line_stripped.lower().startswith("chapter"):
                        if isinstance(current_chapter["prompt"], list):
                            current_chapter["prompt"].append(f"- {line_stripped}")

            # Add the last chapter if it exists
            if current_chapter and current_chapter.get("prompt"):
                if isinstance(current_chapter["prompt"], list):
                    current_chapter["prompt"] = "\n".join(current_chapter["prompt"])
                chapters.append(current_chapter)
                current_chapter = None

        if not chapters:
            logger.error("Emergency processing failed to find any chapters")
            # Create placeholder chapters as last resort
            logger.warning("Creating placeholder chapters as fallback")
            for i in range(1, num_chapters + 1):
                chapters.append({
                    "chapter_number": i,
                    "title": f"Chapter {i}",
                    "prompt": (
                        "- Key events: Develop the story forward\n"
                        "- Character developments: Continue character arcs\n"
                        "- Setting: Maintain consistent world\n"
                        "- Tone: Match previous chapters"
                    ),
                })
            return chapters

        # Sort and dedupe
        seen_numbers = set()
        unique_chapters = []
        for ch in sorted(chapters, key=lambda x: x["chapter_number"]):
            if ch["chapter_number"] not in seen_numbers and ch["chapter_number"] <= num_chapters:
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
