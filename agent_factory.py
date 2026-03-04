"""Factory for creating agent group chats - reduces duplication between modules"""
from typing import Any, Dict, List, Optional

import autogen

from constants import GroupChatConstants
from utils import get_logger


logger = get_logger("agent_factory")


class AgentManager:
    """Base class for managing agents and creating group chats"""
    
    def __init__(self, agents: Dict[str, autogen.ConversableAgent]):
        self.agents = agents
        self._chat_history: List[Dict[str, Any]] = []
    
    def get_agent(self, name: str) -> Optional[autogen.ConversableAgent]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all available agent names"""
        return list(self.agents.keys())


class GroupChatFactory:
    """Factory for creating pre-configured group chats"""
    
    @staticmethod
    def create_outline_chat(
        agents: Dict[str, autogen.ConversableAgent],
        max_rounds: int = GroupChatConstants.OUTLINE_MAX_ROUNDS
    ) -> autogen.GroupChat:
        """Create a group chat for outline generation"""
        logger.debug(f"Creating outline group chat with max_rounds={max_rounds}")
        
        required_agents = ["user_proxy", "story_planner", "world_builder", "outline_creator"]
        available_agents = []
        
        for agent_name in required_agents:
            if agent_name in agents:
                available_agents.append(agents[agent_name])
            else:
                logger.warning(f"Required agent '{agent_name}' not found in agents dict")
        
        if len(available_agents) < len(required_agents):
            logger.error(f"Missing required agents for outline chat. Have {len(available_agents)}, need {len(required_agents)}")
        
        return autogen.GroupChat(
            agents=available_agents,
            messages=[],
            max_round=max_rounds,
            speaker_selection_method=GroupChatConstants.SPEAKER_SELECTION
        )
    
    @staticmethod
    def create_chapter_chat(
        agents: Dict[str, autogen.ConversableAgent],
        agent_config: Dict[str, Any],
        outline_context: str,
        max_rounds: int = GroupChatConstants.CHAPTER_MAX_ROUNDS
    ) -> autogen.GroupChat:
        """Create a group chat for chapter generation"""
        logger.debug(f"Creating chapter group chat with max_rounds={max_rounds}")
        
        messages = [{
            "role": "system",
            "content": f"Complete Book Outline:\n{outline_context}"
        }]
        
        # Create a copy of the writer agent for final output
        writer_final = autogen.AssistantAgent(
            name="writer_final",
            system_message=agents["writer"].system_message,
            llm_config=agent_config
        )
        
        return autogen.GroupChat(
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
    def create_retry_chat(
        agents: Dict[str, autogen.ConversableAgent],
        max_rounds: int = GroupChatConstants.REPLY_MAX_ROUNDS
    ) -> autogen.GroupChat:
        """Create a minimal group chat for emergency retry"""
        logger.debug(f"Creating retry group chat with max_rounds={max_rounds}")
        
        return autogen.GroupChat(
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
    """Manages group chats with history tracking"""
    
    def __init__(self, agents: Dict[str, autogen.ConversableAgent]):
        super().__init__(agents)
        self._active_chats: Dict[str, autogen.GroupChat] = {}
        self._factory = GroupChatFactory()
    
    def create_outline_chat(self, max_rounds: int = GroupChatConstants.OUTLINE_MAX_ROUNDS) -> autogen.GroupChat:
        """Create and track an outline generation chat"""
        chat = self._factory.create_outline_chat(self.agents, max_rounds)
        self._active_chats["outline"] = chat
        return chat
    
    def create_chapter_chat(
        self, 
        agent_config: Dict[str, Any],
        outline_context: str,
        max_rounds: int = GroupChatConstants.CHAPTER_MAX_ROUNDS
    ) -> autogen.GroupChat:
        """Create and track a chapter generation chat"""
        chat = self._factory.create_chapter_chat(
            self.agents, agent_config, outline_context, max_rounds
        )
        self._active_chats["chapter"] = chat
        return chat
    
    def create_retry_chat(self, max_rounds: int = GroupChatConstants.REPLY_MAX_ROUNDS) -> autogen.GroupChat:
        """Create and track a retry chat"""
        chat = self._factory.create_retry_chat(self.agents, max_rounds)
        self._active_chats["retry"] = chat
        return chat
    
    def get_chat_history(self, chat_name: str) -> List[Dict[str, Any]]:
        """Get history for a specific chat"""
        if chat_name in self._active_chats:
            return self._active_chats[chat_name].messages
        return []
    
    def clear_chat(self, chat_name: str) -> None:
        """Clear a chat from active chats"""
        if chat_name in self._active_chats:
            del self._active_chats[chat_name]
            logger.debug(f"Cleared chat: {chat_name}")


# Convenience functions for backwards compatibility
def create_outline_groupchat(
    agents: Dict[str, autogen.ConversableAgent],
    max_rounds: int = GroupChatConstants.OUTLINE_MAX_ROUNDS
) -> autogen.GroupChat:
    """Create a group chat for outline generation (backwards compatible)"""
    return GroupChatFactory.create_outline_chat(agents, max_rounds)


def create_chapter_groupchat(
    agents: Dict[str, autogen.ConversableAgent],
    agent_config: Dict[str, Any],
    outline_context: str,
    max_rounds: int = GroupChatConstants.CHAPTER_MAX_ROUNDS
) -> autogen.GroupChat:
    """Create a group chat for chapter generation (backwards compatible)"""
    return GroupChatFactory.create_chapter_chat(
        agents, agent_config, outline_context, max_rounds
    )
