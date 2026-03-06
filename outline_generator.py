"""Generate book outlines using AutoGen agents with improved error handling and validation

Supports both legacy AutoGen 0.2 and AutoGen 2.0 (0.4+) APIs.
AutoGen 2.0 uses async team execution with termination conditions.
"""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional

# AutoGen 2.0 imports
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
AUTOGEN_2_AVAILABLE = True

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
            chapters = self._generate_outline_autogen2(initial_prompt, num_chapters)

            if not chapters:
                raise ParseError("Failed to extract any chapters from outline")

            # Sort and verify sequence
            chapters.sort(key=lambda x: x["chapter_number"])

            # If we got chapters but not enough, log warning and fill with placeholders
            if len(chapters) < num_chapters:
                logger.warning(f"Only extracted {len(chapters)} chapters out of {num_chapters} required")
                # Fill in missing chapters with placeholders
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

            logger.info(f"Successfully generated outline with {len(chapters)} chapters")
            return chapters

        except Exception as e:
            logger.error(f"Failed to generate outline: {e}")
            raise

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
3. Outline Creator: Generate a detailed outline with chapter titles and prompts for ALL {num_chapters} CHAPTERS

CRITICAL REQUIREMENTS:
- You MUST generate ALL {num_chapters} chapters - do not stop early
- Each chapter MUST have exactly these fields: title, key_events, character_developments, setting, tone
- Every chapter MUST have at least {OutlineConstants.MIN_EVENTS_PER_CHAPTER} Key Events
- Total chapters in outline: {num_chapters} (exactly {num_chapters}, no more, no less)
- Do not combine chapters
- Do not leave any chapters undefined
- Think through every chapter carefully
- Number chapters sequentially from 1 to {num_chapters}

IMPORTANT: Provide your response as a JSON array. Start with "OUTLINE:" followed by valid JSON:

OUTLINE:
[
  {{
    "chapter_number": 1,
    "title": "Chapter Title Here",
    "key_events": ["Event 1", "Event 2", "Event 3"],
    "character_developments": "Description of character growth",
    "setting": "Location and atmosphere",
    "tone": "Emotional and narrative tone"
  }},
  {{
    "chapter_number": 2,
    "title": "Chapter 2 Title",
    "key_events": ["Event A", "Event B", "Event C"],
    "character_developments": "Description of character growth",
    "setting": "Location and atmosphere",
    "tone": "Emotional and narrative tone"
  }},
  ... continue for ALL {num_chapters} chapters ...
]

End the outline with '{AgentConstants.OUTLINE_END_TAG}' only after ALL {num_chapters} chapters have been listed in the JSON array."""

    def _process_outline_results(
        self, messages: List[Dict[str, Any]], num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Extract and process the outline with JSON or text format requirements"""
        logger.debug("Processing outline from messages")
        outline_content = self._extract_outline_content(messages)

        if not outline_content:
            logger.warning("No structured outline found")
            return []

        # Try JSON parsing first (more reliable)
        chapters = self._try_parse_json_outline(outline_content, num_chapters)

        # Fall back to regex parsing if JSON fails
        if not chapters:
            logger.debug("JSON parsing failed, trying regex parsing")
            chapters = self._parse_chapter_sections(outline_content, num_chapters)

        if len(chapters) < num_chapters:
            logger.warning(
                f"Only extracted {len(chapters)} chapters out of {num_chapters} required"
            )

        return chapters

    def _try_parse_json_outline(
        self, content: str, num_chapters: int
    ) -> List[Dict[str, Any]]:
        """Try to parse outline as JSON format"""
        json_content = None

        # Strip "OUTLINE:" prefix if present
        if content.startswith("OUTLINE:"):
            content = content[7:].strip()  # Remove "OUTLINE:" prefix

        # Method 1: Look for content between ```json and ``` code blocks
        json_block_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
        if json_block_match:
            json_content = json_block_match.group(1)
            logger.debug("Found JSON in markdown code block")

        # Method 2: Look for JSON array pattern with bracket counting
        if not json_content:
            # Find the first [ and track brackets to find the matching ]
            bracket_start = content.find("[")
            if bracket_start != -1 and '"chapter_number"' in content[bracket_start:bracket_start+1000]:
                bracket_count = 0
                in_string = False
                escape_next = False
                for i in range(bracket_start, len(content)):
                    char = content[i]

                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_content = content[bracket_start:i+1]
                                logger.debug(f"Found JSON array from position {bracket_start} to {i}")
                                break

        if not json_content:
            logger.debug("No valid JSON array found in content")
            return []

        # Clean JSON to handle common LLM errors (trailing commas, etc.)
        json_content = self._clean_json(json_content)

        # Log a preview of the JSON for debugging
        logger.debug(f"JSON content preview (first 500 chars): {json_content[:500]}")

        try:
            logger.debug("Attempting to parse outline as JSON")
            data = json.loads(json_content)

            if not isinstance(data, list):
                logger.warning(f"JSON outline is not a list: {type(data)}")
                return []

            chapters = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                # Extract fields with fallbacks
                chapter_num = item.get("chapter_number", len(chapters) + 1)
                title = item.get("title", f"Chapter {chapter_num}")

                # Build prompt from JSON fields
                key_events = item.get("key_events", [])
                if isinstance(key_events, list):
                    events_text = "\n".join(f"- {event}" for event in key_events)
                else:
                    events_text = str(key_events)

                character_dev = item.get("character_developments", "Continue character arcs from previous chapters")
                setting = item.get("setting", "Maintain consistent world setting")
                tone = item.get("tone", "Match the established narrative tone")

                prompt_parts = [
                    f"- Key events: {events_text}",
                    f"- Character developments: {character_dev}",
                    f"- Setting: {setting}",
                    f"- Tone: {tone}"
                ]

                chapters.append({
                    "chapter_number": chapter_num,
                    "title": title,
                    "prompt": "\n".join(prompt_parts),
                })

            logger.info(f"Successfully parsed {len(chapters)} chapters from JSON")
            return chapters

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON outline: {e}")
            logger.debug(f"JSON content that failed to parse: {json_content[:500]}")
            return []
        except Exception as e:
            logger.warning(f"Error processing JSON outline: {e}")
            return []

    def _clean_json(self, json_str: str) -> str:
        """Clean JSON string to handle common LLM generation errors"""
        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        # Remove trailing commas inside strings would be more complex, but the above handles most cases
        return json_str

    def _extract_outline_content(self, messages: List[Dict[str, Any]]) -> str:
        """Extract outline content from messages - supports both JSON and text format"""
        logger.debug(f"Searching {len(messages)} messages for outline content")

        # Debug: Log all message senders to understand the conversation flow
        sender_counts = {}
        for msg in messages:
            sender = msg.get("name", msg.get("role", msg.get("sender", "unknown")))
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        logger.debug(f"Message senders: {sender_counts}")

        # First, look for content between "OUTLINE:" and "END OF OUTLINE" (supports both JSON and text)
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

        # Second, look for JSON in markdown code blocks (```json ... ```)
        for msg in reversed(messages):
            content = msg.get("content", "")
            if '```json' in content and '"chapter_number"' in content:
                # Extract JSON from code block
                json_start = content.find('```json')
                if json_start != -1:
                    json_content_start = json_start + 7  # Skip past ```json
                    json_end = content.find('```', json_content_start)
                    if json_end != -1:
                        json_content = content[json_content_start:json_end].strip()
                        logger.debug(f"Extracted JSON from code block, length: {len(json_content)}")
                        return "OUTLINE:\n" + json_content

        # Third, look for JSON array pattern (with chapter_number field)
        for msg in reversed(messages):
            content = msg.get("content", "")
            if '"chapter_number"' in content or '"title"' in content:
                # Found potential JSON content
                # Look for JSON array start
                bracket_start = content.find("[")
                if bracket_start != -1:
                    # Find the matching closing bracket
                    bracket_count = 0
                    for i in range(bracket_start, len(content)):
                        if content[i] == '[':
                            bracket_count += 1
                        elif content[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                # Found complete JSON array
                                json_content = content[bracket_start:i+1]
                                # Validate it looks like chapter data
                                if '"chapter_number"' in json_content or '"title"' in json_content:
                                    logger.debug(f"Extracted JSON array, length: {len(json_content)}")
                                    return "OUTLINE:\n" + json_content

        # Fourth fallback: look for traditional text chapter markers
        for msg in reversed(messages):
            content = msg.get("content", "")
            if "Chapter 1:" in content or "**Chapter 1:**" in content:
                return content

        # Fifth fallback: look for "End of Chapter X" pattern (narrative content)
        for msg in reversed(messages):
            content = msg.get("content", "")
            if "End of Chapter" in content:
                logger.warning("Detected narrative content instead of outline format")
                return content

        logger.warning("No outline content found in any messages")
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
