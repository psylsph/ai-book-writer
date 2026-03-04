"""Define the agents used in the book generation system with improved context management"""
from typing import Any, Dict, List, Optional, Union

import autogen

from constants import AgentConstants, ChapterConstants, ConfigConstants
from models import Chapter
from utils import get_logger

logger = get_logger("agents")


class BookAgents:
    """Manages creation and configuration of AutoGen agents for book generation"""

    def __init__(
        self,
        agent_config: Dict[str, Any],
        outline: Optional[List[Dict[str, Any]]] = None,
        num_chapters: Optional[int] = None,
        initial_prompt: Optional[str] = None,
    ):
        """Initialize agents with book outline context
        
        Args:
            agent_config: Configuration for AutoGen agents
            outline: Book outline with chapter information (for book generation phase)
            num_chapters: Total number of chapters
            initial_prompt: Initial story premise (for outline generation phase)
        """
        self.agent_config = agent_config
        self.outline = outline
        self.num_chapters = num_chapters or ChapterConstants.DEFAULT_NUM_CHAPTERS
        self.initial_prompt = initial_prompt or ""
        self.world_elements: Dict[str, str] = {}
        self.character_developments: Dict[str, List[str]] = {}
        logger.debug("BookAgents initialized")

    def create_agents(
        self, initial_prompt: Optional[str] = None, num_chapters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create and return all agents needed for book generation"""
        if num_chapters:
            self.num_chapters = num_chapters
        
        outline_context = self._format_outline_context()
        agents: Dict[str, autogen.ConversableAgent] = {}

        # Memory Keeper: Maintains story continuity and context
        agents["memory_keeper"] = self._create_memory_keeper(outline_context)
        logger.debug("Created memory_keeper agent")

        # Story Planner - Focuses on high-level story structure
        agents["story_planner"] = self._create_story_planner()
        logger.debug("Created story_planner agent")

        # Outline Creator - Creates detailed chapter outlines
        agents["outline_creator"] = self._create_outline_creator(
            self.num_chapters, initial_prompt or ""
        )
        logger.debug("Created outline_creator agent")

        # World Builder: Creates and maintains the story setting
        agents["world_builder"] = self._create_world_builder(outline_context)
        logger.debug("Created world_builder agent")

        # Writer: Generates the actual prose
        agents["writer"] = self._create_writer(outline_context)
        logger.debug("Created writer agent")

        # Editor: Reviews and improves content
        agents["editor"] = self._create_editor(outline_context)
        logger.debug("Created editor agent")

        # User Proxy: Manages the interaction
        agents["user_proxy"] = self._create_user_proxy()
        logger.debug("Created user_proxy agent")

        logger.info(f"Created {len(agents)} agents successfully")
        return agents

    def _format_outline_context(self) -> str:
        """Format the book outline into a readable context"""
        if not self.outline:
            return ""

        lines = ["Complete Book Outline:"]
        for chapter in self.outline:
            lines.extend([
                f"\nChapter {chapter['chapter_number']}: {chapter['title']}",
                chapter['prompt']
            ])
        return "\n".join(lines)

    def _create_memory_keeper(self, outline_context: str) -> autogen.AssistantAgent:
        """Create the Memory Keeper agent"""
        # Use initial prompt if no outline context available (during outline generation)
        context = outline_context if outline_context else f"Story Premise:\n{self.initial_prompt}"
        
        return autogen.AssistantAgent(
            name="memory_keeper",
            system_message=f"""You are the keeper of the story's continuity and context.
            
Your responsibilities:
1. Track and summarize each chapter's key events
2. Monitor character development and relationships
3. Maintain world-building consistency
4. Flag any continuity issues

Story Context:
{context}

Format your responses as follows:
- Start updates with '{AgentConstants.MEMORY_UPDATE_TAG}'
- List key events with '{AgentConstants.EVENT_TAG}'
- List character developments with '{AgentConstants.CHARACTER_TAG}'
- List world details with '{AgentConstants.WORLD_TAG}'
- Flag issues with '{AgentConstants.CONTINUITY_ALERT_TAG}'""",
            llm_config=self.agent_config,
        )

    def _create_story_planner(self) -> autogen.AssistantAgent:
        """Create the Story Planner agent"""
        return autogen.AssistantAgent(
            name="story_planner",
            system_message=f"""You are an expert story arc planner focused on overall narrative structure.

Your sole responsibility is creating the high-level story arc.
When given an initial story premise:
1. Identify major plot points and story beats
2. Map character arcs and development
3. Note major story transitions
4. Plan narrative pacing

Format your output EXACTLY as:
{AgentConstants.STORY_ARC_TAG}
- Major Plot Points:
[List each major event that drives the story]

- Character Arcs:
[For each main character, describe their development path]

- Story Beats:
[List key emotional and narrative moments in sequence]

- Key Transitions:
[Describe major shifts in story direction or tone]

Always provide specific, detailed content - never use placeholders.""",
            llm_config=self.agent_config,
        )

    def _create_outline_creator(
        self, num_chapters: int, initial_prompt: str
    ) -> autogen.AssistantAgent:
        """Create the Outline Creator agent"""
        return autogen.AssistantAgent(
            name="outline_creator",
            system_message=f"""Generate a detailed {num_chapters}-chapter outline.

YOU MUST USE EXACTLY THIS FORMAT FOR EACH CHAPTER - NO DEVIATIONS:

Chapter 1: [Title]
Chapter Title: [Same title as above]
Key Events:
- [Event 1]
- [Event 2]
- [Event 3]
Character Developments: [Specific character moments and changes]
Setting: [Specific location and atmosphere]
Tone: [Specific emotional and narrative tone]

[REPEAT THIS EXACT FORMAT FOR ALL {num_chapters} CHAPTERS]

Requirements:
1. EVERY field must be present for EVERY chapter
2. EVERY chapter must have AT LEAST 3 specific Key Events
3. ALL chapters must be detailed - no placeholders
4. Format must match EXACTLY - including all headings and bullet points

Initial Premise:
{initial_prompt}

START WITH '{AgentConstants.OUTLINE_START_TAG}' AND END WITH '{AgentConstants.OUTLINE_END_TAG}'
""",
            llm_config=self.agent_config,
        )

    def _create_world_builder(self, outline_context: str) -> autogen.AssistantAgent:
        """Create the World Builder agent"""
        # Use initial prompt if no outline context available (during outline generation)
        context = outline_context if outline_context else f"Story Premise:\n{self.initial_prompt}"
        
        return autogen.AssistantAgent(
            name="world_builder",
            system_message=f"""You are an expert in world-building who creates rich, consistent settings.

Your role is to establish ALL settings and locations needed for the entire story.

Story Context:
{context}

Your responsibilities:
1. Review the story arc to identify every location and setting needed
2. Create detailed descriptions for each setting, including:
   - Physical layout and appearance
   - Atmosphere and environmental details
   - Important objects or features
   - Sensory details (sights, sounds, smells)
3. Identify recurring locations that appear multiple times
4. Note how settings might change over time
5. Create a cohesive world that supports the story's themes

Format your response as:
{AgentConstants.WORLD_TAG}ELEMENTS:

[LOCATION NAME]:
- Physical Description: [detailed description]
- Atmosphere: [mood, time of day, lighting, etc.]
- Key Features: [important objects, layout elements]
- Sensory Details: [what characters would experience]

[RECURRING ELEMENTS]:
- List any settings that appear multiple times
- Note any changes to settings over time

[TRANSITIONS]:
- How settings connect to each other
- How characters move between locations""",
            llm_config=self.agent_config,
        )

    def _create_writer(self, outline_context: str) -> autogen.AssistantAgent:
        """Create the Writer agent"""
        min_words = ChapterConstants.MIN_WORD_COUNT
        return autogen.AssistantAgent(
            name="writer",
            system_message=f"""You are an expert creative writer who brings scenes to life.

Book Context:
{outline_context}

Your focus:
1. Write according to the outlined plot points
2. Maintain consistent character voices
3. Incorporate world-building details
4. Create engaging prose
5. Please make sure that you write the complete scene, do not leave it incomplete
6. Each chapter MUST be at least {min_words} words (approximately 30,000 characters). Consider this a hard requirement. If your output is shorter, continue writing until you reach this minimum length
7. Ensure transitions are smooth and logical
8. Do not cut off the scene, make sure it has a proper ending
9. Add a lot of details, and describe the environment and characters where it makes sense

Always reference the outline and previous content.
Mark drafts with '{AgentConstants.SCENE_TAG}' and final versions with '{AgentConstants.SCENE_FINAL_TAG}'""",
            llm_config=self.agent_config,
        )

    def _create_editor(self, outline_context: str) -> autogen.AssistantAgent:
        """Create the Editor agent"""
        min_words = ChapterConstants.MIN_WORD_COUNT
        return autogen.AssistantAgent(
            name="editor",
            system_message=f"""You are an expert editor ensuring quality and consistency.

Book Overview:
{outline_context}

Your focus:
1. Check alignment with outline
2. Verify character consistency
3. Maintain world-building rules
4. Improve prose quality
5. Return complete edited chapter
6. Never ask to start the next chapter, as the next step is finalizing this chapter
7. Each chapter MUST be at least {min_words} words. If the content is shorter, return it to the writer for expansion. This is a hard requirement - do not approve chapters shorter than {min_words} words

Format your responses:
1. Start critiques with '{AgentConstants.FEEDBACK_TAG}'
2. Provide suggestions with '{AgentConstants.SUGGEST_TAG}'
3. Return full edited chapter with '{AgentConstants.EDITED_SCENE_TAG}:START' to '{AgentConstants.EDITED_SCENE_TAG}:END'

Reference specific outline elements in your feedback.""",
            llm_config=self.agent_config,
        )

    def _create_user_proxy(self) -> autogen.UserProxyAgent:
        """Create the User Proxy agent"""
        return autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="TERMINATE",
            code_execution_config={
                "work_dir": "book_output",
                "use_docker": False
            }
        )

    def update_world_element(self, element_name: str, description: str) -> None:
        """Track a new or updated world element"""
        self.world_elements[element_name] = description
        logger.debug(f"Updated world element: {element_name}")

    def update_character_development(
        self, character_name: str, development: str
    ) -> None:
        """Track character development"""
        if character_name not in self.character_developments:
            self.character_developments[character_name] = []
        self.character_developments[character_name].append(development)
        logger.debug(f"Updated character development: {character_name}")

    def get_world_context(self) -> str:
        """Get formatted world-building context"""
        if not self.world_elements:
            return "No established world elements yet."

        lines = ["Established World Elements:"]
        for name, desc in self.world_elements.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def get_character_context(self) -> str:
        """Get formatted character development context"""
        if not self.character_developments:
            return "No character developments tracked yet."

        lines = ["Character Development History:"]
        for name, devs in self.character_developments.items():
            lines.append(f"- {name}:")
            for dev in devs:
                lines.append(f"  - {dev}")
        return "\n".join(lines)
