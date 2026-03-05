"""Factory for creating agent group chats - supports both legacy AutoGen 0.2 and AutoGen 2.0 (0.4+)

AutoGen 2.0 uses team-based collaboration patterns instead of GroupChat/GroupChatManager.
Key migration changes:
- GroupChat -> RoundRobinGroupChat or SelectorGroupChat
- GroupChatManager -> Team with termination conditions
- max_round -> MaxMessageTermination condition
"""
from typing import Any, Dict, List, Optional

# AutoGen 2.0 imports
try:
    from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.agents import AssistantAgent
    AUTOGEN_2_AVAILABLE = True
except ImportError:
    AUTOGEN_2_AVAILABLE = False
    RoundRobinGroupChat = None
    SelectorGroupChat = None
    MaxMessageTermination = None
    TextMentionTermination = None
    AssistantAgent = None

# Legacy AutoGen imports
try:
    import autogen as autogen_legacy
    AUTOGEN_LEGACY_AVAILABLE = True
except ImportError:
    AUTOGEN_LEGACY_AVAILABLE = False
    autogen_legacy = None

from constants import GroupChatConstants
from utils import get_logger

logger = get_logger("agent_factory")


class AgentManager:
    """Base class for managing agents and creating group chats/teams"""

    def __init__(self, agents: Dict[str, Any], use_autogen2: bool = True):
        self.agents = agents
        self._chat_history: List[Dict[str, Any]] = []
        self.use_autogen2 = use_autogen2 and AUTOGEN_2_AVAILABLE

    def get_agent(self, name: str) -> Optional[Any]:
        """Get an agent by name"""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all available agent names"""
        return list(self.agents.keys())


class TeamFactory:
    """Factory for creating pre-configured teams (AutoGen 2.0) or group chats (legacy)"""

    @staticmethod
    def create_outline_team(
        agents: Dict[str, Any],
        max_rounds: int = GroupChatConstants.OUTLINE_MAX_ROUNDS,
        use_autogen2: bool = True,
    ) -> Any:
        """Create a team/group chat for outline generation"""
        logger.debug(f"Creating outline team with max_rounds={max_rounds}, AutoGen 2.0: {use_autogen2}")

        required_agents = ["user_proxy", "story_planner", "world_builder", "outline_creator"]
        available_agents = []

        for agent_name in required_agents:
            if agent_name in agents:
                available_agents.append(agents[agent_name])
            else:
                logger.warning(f"Required agent '{agent_name}' not found in agents dict")

        if len(available_agents) < len(required_agents):
            logger.error(f"Missing required agents for outline chat. Have {len(available_agents)}, need {len(required_agents)}")

        if use_autogen2 and AUTOGEN_2_AVAILABLE:
            return TeamFactory._create_outline_team_autogen2(available_agents, max_rounds)
        else:
            return TeamFactory._create_outline_groupchat_legacy(available_agents, max_rounds)

    @staticmethod
    def _create_outline_team_autogen2(agents: List[Any], max_rounds: int) -> Any:
        """Create a RoundRobinGroupChat for outline generation (AutoGen 2.0)"""
        logger.debug(f"Creating AutoGen 2.0 RoundRobinGroupChat for outline generation")

        if RoundRobinGroupChat is None or MaxMessageTermination is None:
            raise ImportError("AutoGen 2.0 is not available. Install with: pip install 'autogen-agentchat>=0.4.0' 'autogen-ext[openai]>=0.4.0'")

        # Create termination condition
        termination = MaxMessageTermination(max_rounds)

        # Create the team
        team = RoundRobinGroupChat(
            participants=agents,
            termination_condition=termination,
        )

        return team

    @staticmethod
    def _create_outline_groupchat_legacy(agents: List[Any], max_rounds: int) -> Any:
        """Create a GroupChat for outline generation (legacy AutoGen)"""
        if not AUTOGEN_LEGACY_AVAILABLE:
            raise ImportError("Legacy AutoGen not available. Install with: pip install pyautogen")

        return autogen_legacy.GroupChat(
            agents=agents,
            messages=[],
            max_round=max_rounds,
            speaker_selection_method=GroupChatConstants.SPEAKER_SELECTION,
        )

    @staticmethod
    def create_chapter_team(
        agents: Dict[str, Any],
        agent_config: Dict[str, Any],
        outline_context: str,
        max_rounds: int = GroupChatConstants.CHAPTER_MAX_ROUNDS,
        use_autogen2: bool = True,
    ) -> Any:
        """Create a team/group chat for chapter generation"""
        logger.debug(f"Creating chapter team with max_rounds={max_rounds}, AutoGen 2.0: {use_autogen2}")

        if use_autogen2 and AUTOGEN_2_AVAILABLE:
            return TeamFactory._create_chapter_team_autogen2(agents, outline_context, max_rounds)
        else:
            return TeamFactory._create_chapter_groupchat_legacy(agents, agent_config, outline_context, max_rounds)

    @staticmethod
    def _create_chapter_team_autogen2(
        agents: Dict[str, Any],
        outline_context: str,
        max_rounds: int,
    ) -> Any:
        """Create a RoundRobinGroupChat for chapter generation (AutoGen 2.0)"""
        logger.debug(f"Creating AutoGen 2.0 RoundRobinGroupChat for chapter generation")

        if RoundRobinGroupChat is None or MaxMessageTermination is None or AssistantAgent is None:
            raise ImportError("AutoGen 2.0 is not available. Install with: pip install 'autogen-agentchat>=0.4.0' 'autogen-ext[openai]>=0.4.0'")

        # Create writer_final agent (clone of writer with same config)
        writer = agents.get("writer")
        if writer is None:
            raise ValueError("Writer agent not found in agents dict")

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
            agent = agents.get(agent_name)
            if agent is not None:
                participants.append(agent)

        # Add writer_final at the end
        participants.append(writer_final)

        # Create termination condition
        termination = MaxMessageTermination(max_rounds)

        # Create the team with participants
        team = RoundRobinGroupChat(
            participants=participants,
            termination_condition=termination,
        )

        return team

    @staticmethod
    def _create_chapter_groupchat_legacy(
        agents: Dict[str, Any],
        agent_config: Dict[str, Any],
        outline_context: str,
        max_rounds: int,
    ) -> Any:
        """Create a GroupChat for chapter generation (legacy AutoGen)"""
        if not AUTOGEN_LEGACY_AVAILABLE:
            raise ImportError("Legacy AutoGen not available. Install with: pip install pyautogen")

        messages = [{
            "role": "system",
            "content": f"Complete Book Outline:\n{outline_context}"
        }]

        # Create a copy of the writer agent for final output
        writer_final = autogen_legacy.ConversableAgent(
            name="writer_final",
            system_message=agents["writer"].system_message,
            llm_config=agent_config
        )

        return autogen_legacy.GroupChat(
            agents=[
                agents["user_proxy"],
                agents["memory_keeper"],
                agents["writer"],
                agents["editor"],
                writer_final
            ],
            messages=messages,
            max_round=max_rounds,
            speaker_selection_method=GroupChatConstants.SPEAKER_SELECTION
        )

    @staticmethod
    def create_retry_team(
        agents: Dict[str, Any],
        max_rounds: int = GroupChatConstants.REPLY_MAX_ROUNDS,
        use_autogen2: bool = True,
    ) -> Any:
        """Create a minimal team/group chat for emergency retry"""
        logger.debug(f"Creating retry team with max_rounds={max_rounds}, AutoGen 2.0: {use_autogen2}")

        if use_autogen2 and AUTOGEN_2_AVAILABLE:
            return TeamFactory._create_retry_team_autogen2(agents, max_rounds)
        else:
            return TeamFactory._create_retry_groupchat_legacy(agents, max_rounds)

    @staticmethod
    def _create_retry_team_autogen2(agents: Dict[str, Any], max_rounds: int) -> Any:
        """Create a minimal RoundRobinGroupChat for emergency retry (AutoGen 2.0)"""
        logger.debug(f"Creating AutoGen 2.0 RoundRobinGroupChat for emergency retry")

        if RoundRobinGroupChat is None or MaxMessageTermination is None:
            raise ImportError("AutoGen 2.0 is not available. Install with: pip install 'autogen-agentchat>=0.4.0' 'autogen-ext[openai]>=0.4.0'")

        # Build participants list, ensuring no None values
        available_agents = []
        for agent_name in ["user_proxy", "writer"]:
            agent = agents.get(agent_name)
            if agent is not None:
                available_agents.append(agent)

        # Add memory_keeper if available
        if "memory_keeper" in agents:
            agent = agents.get("memory_keeper")
            if agent is not None:
                available_agents.insert(1, agent)

        # Create termination condition
        termination = MaxMessageTermination(max_rounds)

        # Create the team
        team = RoundRobinGroupChat(
            participants=available_agents,
            termination_condition=termination,
        )

        return team

    @staticmethod
    def _create_retry_groupchat_legacy(agents: Dict[str, Any], max_rounds: int) -> Any:
        """Create a minimal GroupChat for emergency retry (legacy AutoGen)"""
        if not AUTOGEN_LEGACY_AVAILABLE:
            raise ImportError("Legacy AutoGen not available. Install with: pip install pyautogen")

        return autogen_legacy.GroupChat(
            agents=[
                agents["user_proxy"],
                agents["story_planner"],
                agents["writer"]
            ],
            messages=[],
            max_round=max_rounds,
            speaker_selection_method="round_robin"
        )


class ChatManager(AgentManager):
    """Manages group chats/teams with history tracking"""

    def __init__(self, agents: Dict[str, Any], use_autogen2: bool = True):
        super().__init__(agents, use_autogen2)
        self._active_chats: Dict[str, Any] = {}
        self._factory = TeamFactory()

    def create_outline_chat(self, max_rounds: int = GroupChatConstants.OUTLINE_MAX_ROUNDS) -> Any:
        """Create and track an outline generation chat/team"""
        chat = self._factory.create_outline_team(
            self.agents,
            max_rounds,
            use_autogen2=self.use_autogen2,
        )
        self._active_chats["outline"] = chat
        return chat

    def create_chapter_chat(
        self,
        agent_config: Dict[str, Any],
        outline_context: str,
        max_rounds: int = GroupChatConstants.CHAPTER_MAX_ROUNDS,
    ) -> Any:
        """Create and track a chapter generation chat/team"""
        chat = self._factory.create_chapter_team(
            self.agents,
            agent_config,
            outline_context,
            max_rounds,
            use_autogen2=self.use_autogen2,
        )
        self._active_chats["chapter"] = chat
        return chat

    def create_retry_chat(self, max_rounds: int = GroupChatConstants.REPLY_MAX_ROUNDS) -> Any:
        """Create and track a retry chat/team"""
        chat = self._factory.create_retry_team(
            self.agents,
            max_rounds,
            use_autogen2=self.use_autogen2,
        )
        self._active_chats["retry"] = chat
        return chat

    def get_chat_history(self, chat_name: str) -> List[Dict[str, Any]]:
        """Get history for a specific chat"""
        if chat_name in self._active_chats:
            chat = self._active_chats[chat_name]
            # AutoGen 2.0 teams don't have messages attribute, need to handle differently
            if hasattr(chat, "messages"):
                return chat.messages
            # For AutoGen 2.0, return empty list (history would need to be tracked separately)
            return []
        return []

    def clear_chat(self, chat_name: str) -> None:
        """Clear a chat from active chats"""
        if chat_name in self._active_chats:
            del self._active_chats[chat_name]
            logger.debug(f"Cleared chat: {chat_name}")


# Convenience functions for backwards compatibility
def create_outline_groupchat(
    agents: Dict[str, Any],
    max_rounds: int = GroupChatConstants.OUTLINE_MAX_ROUNDS,
    use_autogen2: bool = True,
) -> Any:
    """Create a group chat/team for outline generation (backwards compatible)"""
    return TeamFactory.create_outline_team(agents, max_rounds, use_autogen2)


def create_chapter_groupchat(
    agents: Dict[str, Any],
    agent_config: Dict[str, Any],
    outline_context: str,
    max_rounds: int = GroupChatConstants.CHAPTER_MAX_ROUNDS,
    use_autogen2: bool = True,
) -> Any:
    """Create a group chat/team for chapter generation (backwards compatible)"""
    return TeamFactory.create_chapter_team(
        agents, agent_config, outline_context, max_rounds, use_autogen2
    )
