"""Configuration for the book generation system with environment variable support"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from constants import ConfigConstants, LoggingConstants
from exceptions import ConfigurationError

# AutoGen 2.0 imports
try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelInfo, ModelFamily
    from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
    from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.base import Response
    AUTOGEN_2_AVAILABLE = True
except ImportError:
    AUTOGEN_2_AVAILABLE = False
    OpenAIChatCompletionClient = None
    ModelInfo = None
    ModelFamily = None
    AssistantAgent = None
    CodeExecutorAgent = None
    RoundRobinGroupChat = None
    SelectorGroupChat = None
    MaxMessageTermination = None
    TextMentionTermination = None
    Response = None

# Legacy AutoGen imports for backward compatibility during transition
try:
    import autogen as autogen_legacy
    AUTOGEN_LEGACY_AVAILABLE = True
except ImportError:
    AUTOGEN_LEGACY_AVAILABLE = False
    autogen_legacy = None


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider"""
    name: str
    model: str
    base_url: str
    api_key: str
    timeout: int = ConfigConstants.DEFAULT_TIMEOUT
    max_tokens: Optional[int] = None
    # Price per 1K tokens: [prompt_price, completion_price]
    price: Optional[List[float]] = None  # e.g., [0.03, 0.06] for GPT-4
    # Determine if this is an external (remote) or internal (local) provider
    is_external: bool = False
    
    def get_price_config(self) -> Optional[List[float]]:
        """Get price configuration for cost tracking"""
        return self.price


@dataclass
class AgentConfig:
    """Configuration for AutoGen agents - supports both legacy and AutoGen 2.0"""
    seed: int = ConfigConstants.DEFAULT_SEED
    temperature: float = ConfigConstants.DEFAULT_TEMPERATURE
    timeout: int = ConfigConstants.DEFAULT_TIMEOUT
    cache_seed: Optional[int] = None
    config_list: List[Dict] = field(default_factory=list)
    # AutoGen 2.0 specific fields
    model_client: Optional[OpenAIChatCompletionClient] = None
    model_info: Optional[ModelInfo] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for legacy AutoGen"""
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "config_list": self.config_list,
            "timeout": self.timeout,
            "cache_seed": self.cache_seed
        }

    def get_model_client(self) -> Optional[OpenAIChatCompletionClient]:
        """Get the model client for AutoGen 2.0"""
        if self.model_client:
            return self.model_client

        # Create model client from config_list if available
        if self.config_list and AUTOGEN_2_AVAILABLE:
            config = self.config_list[0]  # Use first config
            return OpenAIChatCompletionClient(
                model=config.get("model", "gpt-4"),
                base_url=config.get("base_url"),
                api_key=config.get("api_key", "")
            )
        return None

    def get_model_info(self) -> Optional[ModelInfo]:
        """Get model info for AutoGen 2.0"""
        if self.model_info:
            return self.model_info

        if self.config_list and AUTOGEN_2_AVAILABLE and ModelInfo:
            config = self.config_list[0]
            # Extract pricing if available
            price = config.get("price")
            if price and isinstance(price, list) and len(price) == 2:
                return ModelInfo(
                    model=config.get("model", "gpt-4"),
                    price_prompt=price[0],
                    price_completion=price[1]
                )
        return None


@dataclass
class AppConfig:
    """Main application configuration"""
    # LLM Provider settings
    provider: str = "local"  # "local", "openai", "azure"
    
    # Local LLM settings
    local_model: str = ConfigConstants.DEFAULT_MODEL
    local_url: str = ConfigConstants.DEFAULT_BASE_URL
    local_api_key: str = ConfigConstants.DEFAULT_API_KEY
    
    # Optional: Separate models for different agent types (local provider only)
    # Creative model - for Writer agent
    local_creative_model: Optional[str] = None
    local_creative_url: Optional[str] = None
    local_creative_temperature: float = ConfigConstants.DEFAULT_TEMPERATURE
    
    # Planning/Review model - for Story Planner, World Builder, Memory Keeper, Editor, Outline Creator
    local_planning_model: Optional[str] = None
    local_planning_url: Optional[str] = None
    local_planning_temperature: float = ConfigConstants.DEFAULT_TEMPERATURE
    
    # OpenAI settings (from environment)
    openai_model: str = "gpt-4"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # OpenAI Creative settings (for writer agent - can use different provider)
    openai_creative_api_key: Optional[str] = None
    openai_creative_model: Optional[str] = None
    openai_creative_base_url: Optional[str] = None
    
    # Azure settings (from environment)
    azure_deployment: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_base_url: Optional[str] = None
    azure_api_version: str = "2024-02-01"
    
    # Output settings
    output_dir: str = "book_output"
    
    # Generation settings
    default_num_chapters: int = 25
    min_word_count: int = 3000
    max_retries: int = 3
    
    # Logging settings
    log_level: str = LoggingConstants.DEFAULT_LOG_LEVEL
    
    # Caching settings
    enable_caching: bool = False  # Set to True to enable LLM response caching
    cache_seed: int = 42  # Seed for cache (used when enable_caching=True)
    
    # Pricing settings (for cost tracking with AutoGen)
    # Format: [prompt_price_per_1k, completion_price_per_1k] in USD
    # Example for GPT-4: [0.03, 0.06]
    # Example for GPT-3.5: [0.0015, 0.002]
    price_prompt_per_1k: Optional[float] = None
    price_completion_per_1k: Optional[float] = None

    # Max tokens settings for different model types
    # Internal models (local LLMs) - typically smaller context windows
    internal_max_tokens: int = ConfigConstants.INTERNAL_MAX_TOKENS
    # External models (OpenAI, etc.) - typically larger context windows
    external_max_tokens: int = ConfigConstants.EXTERNAL_MAX_TOKENS

    # Emergency generation settings
    # When enabled, attempts simplified retry if normal generation fails
    # When disabled (default), fails fast to allow debugging the root cause
    emergency_generation_enabled: bool = ConfigConstants.EMERGENCY_GENERATION_ENABLED

    def __post_init__(self):
        """Load configuration from environment variables"""
        # OpenAI settings
        openai_model_env = os.getenv("OPENAI_MODEL")
        if openai_model_env:
            self.openai_model = openai_model_env
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        
        # OpenAI Creative settings (for writer agent)
        self.openai_creative_api_key = os.getenv("OPENAI_CREATIVE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.openai_creative_model = os.getenv("OPENAI_CREATIVE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4"
        self.openai_creative_base_url = os.getenv("OPENAI_CREATIVE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        # Azure settings
        self.azure_deployment = os.getenv("AZURE_DEPLOYMENT")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        self.azure_base_url = os.getenv("AZURE_BASE_URL")
        
        azure_api_version_env = os.getenv("AZURE_API_VERSION")
        if azure_api_version_env:
            self.azure_api_version = azure_api_version_env
        
        # Override from environment if specified
        output_dir_env = os.getenv("BOOK_OUTPUT_DIR")
        if output_dir_env:
            self.output_dir = output_dir_env
            
        log_level_env = os.getenv("BOOK_LOG_LEVEL")
        if log_level_env:
            self.log_level = log_level_env
            
        num_chapters_env = os.getenv("BOOK_NUM_CHAPTERS")
        if num_chapters_env:
            self.default_num_chapters = int(num_chapters_env)
            
        min_words_env = os.getenv("BOOK_MIN_WORDS")
        if min_words_env:
            self.min_word_count = int(min_words_env)
        
        # Local LLM settings
        if os.getenv("LOCAL_MODEL"):
            self.local_model = os.getenv("LOCAL_MODEL")
        if os.getenv("LOCAL_URL"):
            self.local_url = os.getenv("LOCAL_URL")
        if os.getenv("LOCAL_API_KEY"):
            self.local_api_key = os.getenv("LOCAL_API_KEY")
        
        # Dual model configuration for local provider
        # Default to using two-model system (both default to local_model if not specified)
        self.local_creative_model = os.getenv("LOCAL_CREATIVE_MODEL") or self.local_model
        self.local_creative_url = os.getenv("LOCAL_CREATIVE_URL") or self.local_url
        if os.getenv("LOCAL_CREATIVE_TEMPERATURE"):
            self.local_creative_temperature = float(os.getenv("LOCAL_CREATIVE_TEMPERATURE"))
        
        self.local_planning_model = os.getenv("LOCAL_PLANNING_MODEL") or self.local_model
        self.local_planning_url = os.getenv("LOCAL_PLANNING_URL") or self.local_url
        if os.getenv("LOCAL_PLANNING_TEMPERATURE"):
            self.local_planning_temperature = float(os.getenv("LOCAL_PLANNING_TEMPERATURE"))
        
        # Caching configuration
        cache_env = os.getenv("LLM_CACHE_ENABLED", "false").lower()
        self.enable_caching = cache_env in ("true", "1", "yes", "on")
        if os.getenv("LLM_CACHE_SEED"):
            self.cache_seed = int(os.getenv("LLM_CACHE_SEED"))
        
        # Pricing configuration (for cost tracking)
        if os.getenv("LLM_PRICE_PROMPT_PER_1K"):
            self.price_prompt_per_1k = float(os.getenv("LLM_PRICE_PROMPT_PER_1K"))
        if os.getenv("LLM_PRICE_COMPLETION_PER_1K"):
            self.price_completion_per_1k = float(os.getenv("LLM_PRICE_COMPLETION_PER_1K"))

        # Max tokens configuration
        if os.getenv("LLM_INTERNAL_MAX_TOKENS"):
            self.internal_max_tokens = int(os.getenv("LLM_INTERNAL_MAX_TOKENS"))
        if os.getenv("LLM_EXTERNAL_MAX_TOKENS"):
            self.external_max_tokens = int(os.getenv("LLM_EXTERNAL_MAX_TOKENS"))

        # Emergency generation configuration
        emergency_gen_env = os.getenv("BOOK_EMERGENCY_GENERATION", "false").lower()
        self.emergency_generation_enabled = emergency_gen_env in ("true", "1", "yes", "on")
    
    def get_agent_config(self) -> AgentConfig:
        """Get AutoGen-compatible agent configuration"""
        import logging
        logger = logging.getLogger("config")
        
        provider_config = self._get_provider_config()
        
        # Determine cache seed (None = caching disabled)
        cache_seed_value = self.cache_seed if self.enable_caching else None
        
        if self.enable_caching:
            logger.info(f"LLM caching enabled with seed: {self.cache_seed}")
        else:
            logger.debug("LLM caching disabled")
        
        # Build config dict with optional price
        config_item = {
            "model": provider_config.model,
            "base_url": provider_config.base_url,
            "api_key": provider_config.api_key
        }
        
        # Add price if configured
        if provider_config.price:
            config_item["price"] = provider_config.price
            logger.debug(f"Using custom pricing: ${provider_config.price[0]}/1K prompt, ${provider_config.price[1]}/1K completion")
        
        return AgentConfig(
            seed=ConfigConstants.DEFAULT_SEED,
            temperature=ConfigConstants.DEFAULT_TEMPERATURE,
            timeout=provider_config.timeout,
            cache_seed=cache_seed_value,
            config_list=[config_item]
        )
    
    def get_agent_config_for_role(self, role: str) -> AgentConfig:
        """Get configuration for a specific agent role with dual model support
        
        Args:
            role: The agent role ("writer", "editor", "story_planner", etc.)
            
        Returns:
            AgentConfig appropriate for that role
        """
        import logging
        logger = logging.getLogger("config")
        
        # Determine which model to use based on role
        is_creative_role = role in ["writer"]
        is_planning_role = role in ["story_planner", "world_builder", "memory_keeper", 
                                    "editor", "outline_creator"]
        
        # Determine cache seed (None = caching disabled)
        cache_seed_value = self.cache_seed if self.enable_caching else None
        
        # Build config item with optional pricing
        def build_config_item(model: str, base_url: str, api_key: str) -> Dict:
            item = {
                "model": model,
                "base_url": base_url,
                "api_key": api_key
            }
            price_list = self._get_price_list()
            if price_list:
                item["price"] = price_list
            return item
        
        if is_creative_role:
            # Check if we have remote creative configuration
            if self.openai_creative_api_key:
                # Use remote API for creative tasks
                logger.info(f"Role '{role}' using remote creative model: {self.openai_creative_model}")
                return AgentConfig(
                    seed=ConfigConstants.DEFAULT_SEED,
                    temperature=self.local_creative_temperature,
                    timeout=ConfigConstants.DEFAULT_TIMEOUT,
                    cache_seed=cache_seed_value,
                    config_list=[build_config_item(
                        self.openai_creative_model,
                        self.openai_creative_base_url or "https://api.openai.com/v1",
                        self.openai_creative_api_key
                    )]
                )
            elif self.provider == "local" and self.local_creative_model:
                # Use local creative model
                logger.info(f"Role '{role}' using local creative model: {self.local_creative_model}")
                return AgentConfig(
                    seed=ConfigConstants.DEFAULT_SEED,
                    temperature=self.local_creative_temperature,
                    timeout=ConfigConstants.DEFAULT_TIMEOUT,
                    cache_seed=cache_seed_value,
                    config_list=[build_config_item(
                        self.local_creative_model,
                        self.local_creative_url or self.local_url,
                        self.local_api_key
                    )]
                )
        
        if is_planning_role:
            if self.provider == "local" and self.local_planning_model:
                # Use local planning/review model
                logger.info(f"Role '{role}' using local planning model: {self.local_planning_model}")
                return AgentConfig(
                    seed=ConfigConstants.DEFAULT_SEED,
                    temperature=self.local_planning_temperature,
                    timeout=ConfigConstants.DEFAULT_TIMEOUT,
                    cache_seed=cache_seed_value,
                    config_list=[build_config_item(
                        self.local_planning_model,
                        self.local_planning_url or self.local_url,
                        self.local_api_key
                    )]
                )
            elif self.provider == "openai":
                # Use OpenAI for planning
                logger.info(f"Role '{role}' using OpenAI model: {self.openai_model}")
                return AgentConfig(
                    seed=ConfigConstants.DEFAULT_SEED,
                    temperature=self.local_planning_temperature,
                    timeout=ConfigConstants.DEFAULT_TIMEOUT,
                    cache_seed=cache_seed_value,
                    config_list=[build_config_item(
                        self.openai_model,
                        self.openai_base_url or "https://api.openai.com/v1",
                        self.openai_api_key or ""
                    )]
                )
        
        # Fallback to default configuration
        logger.info(f"Role '{role}' using default model: {self.local_model}")
        return self.get_agent_config()
    
    def _get_price_list(self) -> Optional[List[float]]:
        """Get price list for cost tracking if both prices are configured"""
        if self.price_prompt_per_1k is not None and self.price_completion_per_1k is not None:
            return [self.price_prompt_per_1k, self.price_completion_per_1k]
        return None
    
    def _get_provider_config(self) -> LLMProviderConfig:
        """Get configuration for the selected provider"""
        if self.provider == "openai":
            if not self.openai_api_key:
                raise ConfigurationError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                )
            return LLMProviderConfig(
                name="openai",
                model=self.openai_model,
                base_url=self.openai_base_url or "https://api.openai.com/v1",
                api_key=self.openai_api_key,
                timeout=ConfigConstants.DEFAULT_TIMEOUT
            )
        
        elif self.provider == "azure":
            if not self.azure_api_key or not self.azure_deployment:
                raise ConfigurationError(
                    "Azure configuration incomplete. Set AZURE_API_KEY and AZURE_DEPLOYMENT."
                )
            return LLMProviderConfig(
                name="azure",
                model=self.azure_deployment,
                base_url=f"{self.azure_base_url}/openai/deployments/{self.azure_deployment}",
                api_key=self.azure_api_key,
                timeout=ConfigConstants.DEFAULT_TIMEOUT
            )
        
        else:  # local (default)
            return LLMProviderConfig(
                name="local",
                model=self.local_model,
                base_url=self.local_url,
                api_key=self.local_api_key,
                timeout=ConfigConstants.DEFAULT_TIMEOUT
            )
    
    def validate(self) -> None:
        """Validate configuration values"""
        if self.default_num_chapters < 1:
            raise ConfigurationError("Number of chapters must be at least 1")
        if self.min_word_count < 100:
            raise ConfigurationError("Minimum word count must be at least 100")
        if self.max_retries < 0:
            raise ConfigurationError("Max retries must be non-negative")
        if self.provider not in ["local", "openai", "azure"]:
            raise ConfigurationError(f"Unknown provider: {self.provider}")
    
    # AutoGen 2.0 Model Client Creation Methods
    def create_model_client(self, role: str = "default") -> OpenAIChatCompletionClient:
        """Create an AutoGen 2.0 model client for the specified role

        Args:
            role: Agent role to determine which model configuration to use

        Returns:
            OpenAIChatCompletionClient instance configured for the role

        Raises:
            ConfigurationError: If AutoGen 2.0 is not available or configuration is invalid
        """
        if not AUTOGEN_2_AVAILABLE:
            raise ConfigurationError(
                "AutoGen 2.0 packages not available. "
                "Install with: pip install 'autogen-agentchat>=0.4.0' 'autogen-ext[openai]>=0.4.0'"
            )

        import logging
        logger = logging.getLogger("config")

        # Get provider configuration for the role
        provider_config = self._get_provider_config_for_role(role)

        # Create and return the model client
        try:
            # Build model_info for non-standard models
            model_info = self._create_model_info_for_model(provider_config["model"])

            client = OpenAIChatCompletionClient(
                model=provider_config["model"],
                base_url=provider_config["base_url"],
                api_key=provider_config["api_key"],
                # Add timeout configuration
                timeout=provider_config.get("timeout", ConfigConstants.DEFAULT_TIMEOUT),
                max_tokens=provider_config.get("max_tokens"),
                model_info=model_info,
            )
            logger.info(f"Created model client for role '{role}' with model: {provider_config['model']}, max_tokens: {provider_config.get('max_tokens')}")
            return client
        except Exception as e:
            raise ConfigurationError(f"Failed to create model client for role '{role}': {e}")

    def create_model_client_for_role(self, role: str) -> OpenAIChatCompletionClient:
        """Create a model client specifically for a role with dual-model support

        This is the preferred method for AutoGen 2.0 - it handles the creative/planning
        model split automatically.

        Args:
            role: The agent role ("writer", "editor", "story_planner", etc.)

        Returns:
            OpenAIChatCompletionClient configured for the role

        Raises:
            ConfigurationError: If AutoGen 2.0 is not available
        """
        if not AUTOGEN_2_AVAILABLE:
            raise ConfigurationError(
                "AutoGen 2.0 packages not available. "
                "Install with: pip install 'autogen-agentchat>=0.4.0' 'autogen-ext[openai]>=0.4.0'"
            )

        import logging
        logger = logging.getLogger("config")

        provider_config = self._get_provider_config_for_role(role)

        try:
            # Build model_info for non-standard models
            model_info = self._create_model_info_for_model(provider_config["model"])

            client = OpenAIChatCompletionClient(
                model=provider_config["model"],
                base_url=provider_config["base_url"],
                api_key=provider_config["api_key"],
                timeout=provider_config.get("timeout", ConfigConstants.DEFAULT_TIMEOUT),
                max_tokens=provider_config.get("max_tokens"),
                model_info=model_info,
            )
            logger.info(f"Created model client for role '{role}' with model: {provider_config['model']}, max_tokens: {provider_config.get('max_tokens')}")
            return client
        except Exception as e:
            raise ConfigurationError(f"Failed to create model client for role '{role}': {e}")
    
    def _create_model_info_for_model(self, model_name: str) -> ModelInfo:
        """Create ModelInfo for a given model name.
        
        For standard OpenAI models, returns None to let AutoGen use built-in info.
        For custom/local models, returns appropriate model info.
        """
        if ModelInfo is None:
            return None
            
        # Standard OpenAI models - use built-in info
        standard_models = {"gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-3.5-turbo", 
                          "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"}
        if model_name.lower() in standard_models:
            return None
        
        # For custom/local models, specify capabilities
        # Most modern LLMs support these basic capabilities
        return ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=False,
        )
    
    def _get_provider_config_for_role(self, role: str) -> Dict[str, str]:
        """Get provider configuration for a specific role

        Args:
            role: Agent role (e.g., "writer", "editor", "story_planner")

        Returns:
            Dictionary with model, base_url, api_key, and max_tokens
        """
        # Determine which model to use based on role
        is_creative_role = role in ["writer"]
        is_planning_role = role in ["story_planner", "world_builder", "memory_keeper",
                                    "editor", "outline_creator"]

        if is_creative_role:
            # Check for remote creative configuration first
            if self.openai_creative_api_key:
                return {
                    "model": self.openai_creative_model or "gpt-4",
                    "base_url": self.openai_creative_base_url or "https://api.openai.com/v1",
                    "api_key": self.openai_creative_api_key,
                    "max_tokens": self.external_max_tokens,
                    "is_external": True,
                }
            elif self.provider == "local" and self.local_creative_model:
                return {
                    "model": self.local_creative_model,
                    "base_url": self.local_creative_url or self.local_url,
                    "api_key": self.local_api_key,
                    "max_tokens": self.internal_max_tokens,
                    "is_external": False,
                }

        if is_planning_role:
            if self.provider == "local" and self.local_planning_model:
                return {
                    "model": self.local_planning_model,
                    "base_url": self.local_planning_url or self.local_url,
                    "api_key": self.local_api_key,
                    "max_tokens": self.internal_max_tokens,
                    "is_external": False,
                }
            elif self.provider == "openai" and self.openai_api_key:
                return {
                    "model": self.openai_model,
                    "base_url": self.openai_base_url or "https://api.openai.com/v1",
                    "api_key": self.openai_api_key,
                    "max_tokens": self.external_max_tokens,
                    "is_external": True,
                }

        # Fallback to default configuration
        return {
            "model": self.local_model,
            "base_url": self.local_url,
            "api_key": self.local_api_key,
            "max_tokens": self.internal_max_tokens,
            "is_external": False,
        }


# For backwards compatibility - maintains the old interface
def get_config(local_url: str = ConfigConstants.DEFAULT_BASE_URL) -> Dict:
    """Get the configuration for the agents (legacy interface)
    
    Args:
        local_url: URL for local LLM server (deprecated, use .env file)
        
    Returns:
        Dictionary compatible with AutoGen
    """
    config = AppConfig()
    if local_url != ConfigConstants.DEFAULT_BASE_URL:
        config.local_url = local_url
    
    config.validate()
    agent_config = config.get_agent_config()
    return agent_config.to_dict()


# New preferred way to get configuration
def get_app_config(
    provider: Optional[str] = None,
    output_dir: Optional[str] = None
) -> AppConfig:
    """Get comprehensive application configuration
    
    Args:
        provider: LLM provider to use ("local", "openai", "azure")
        output_dir: Directory for output files
        
    Returns:
        Validated AppConfig instance
    """
    config = AppConfig()
    
    if provider:
        config.provider = provider
    if output_dir:
        config.output_dir = output_dir
    
    config.validate()
    return config


# Example .env file template
ENV_TEMPLATE = """
# LLM Provider Selection
# Options: local, openai, azure
LLM_PROVIDER=local

# OpenAI Configuration (if using openai provider)
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4
# OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

# OpenAI Creative Configuration (for writer agent - can use different endpoint)
# This allows using remote API for creative tasks while using local for planning
# Examples: OpenAI, OpenRouter, or any OpenAI-compatible API
# OPENAI_CREATIVE_API_KEY=your-creative-api-key-here
# OPENAI_CREATIVE_MODEL=gpt-4-turbo-preview
# OPENAI_CREATIVE_BASE_URL=https://api.openai.com/v1  # Or https://openrouter.ai/api/v1

# Azure Configuration (if using azure provider)
AZURE_API_KEY=your-azure-key-here
AZURE_DEPLOYMENT=your-deployment-name
AZURE_BASE_URL=https://your-resource.openai.azure.com
# AZURE_API_VERSION=2024-02-01  # Optional

# Book Generation Settings
BOOK_NUM_CHAPTERS=25
BOOK_MIN_WORDS=3000
BOOK_OUTPUT_DIR=book_output

# Emergency Generation (fallback when normal generation fails)
# When disabled (default), the system will fail fast to allow debugging root causes
# When enabled, attempts a simplified retry if normal generation fails
# Recommended: Keep disabled (false) to focus on fixing the actual problem
BOOK_EMERGENCY_GENERATION=false

# Logging
BOOK_LOG_LEVEL=INFO

# LLM Caching (optional)
# Enable to cache LLM responses and avoid redundant API calls
# Set to 'true' to enable, 'false' to disable (default)
LLM_CACHE_ENABLED=false
# Cache seed - change this to invalidate cache and get fresh responses
LLM_CACHE_SEED=42

# LLM Pricing (optional, for cost tracking)
# Format: price per 1K tokens in USD
# Example for GPT-4: LLM_PRICE_PROMPT_PER_1K=0.03, LLM_PRICE_COMPLETION_PER_1K=0.06
# Example for GPT-3.5: LLM_PRICE_PROMPT_PER_1K=0.0015, LLM_PRICE_COMPLETION_PER_1K=0.002
# LLM_PRICE_PROMPT_PER_1K=0.0
# LLM_PRICE_COMPLETION_PER_1K=0.0

# LLM Max Tokens (optional)
# Maximum tokens for model responses
# For local/internal models (e.g., LM Studio, Ollama)
# LLM_INTERNAL_MAX_TOKENS=32000
# For external/remote models (e.g., OpenAI, Claude)
# LLM_EXTERNAL_MAX_TOKENS=128000
"""


def create_env_template(filepath: str = ".env.example") -> None:
    """Create a template .env file"""
    with open(filepath, "w") as f:
        f.write(ENV_TEMPLATE.strip())


# QMD Configuration Template Extension
QMD_ENV_TEMPLATE = """
# QMD (Query Markup Documents) Integration
# QMD provides local search capabilities across generated chapters
# Install from: https://github.com/tobi/qmd

# Enable QMD integration (requires qmd CLI installed)
QMD_ENABLED=false

# Collection name for book chapters
QMD_COLLECTION_NAME=book_chapters

# Collection name for knowledge base (research materials, world-building, etc.)
QMD_KB_COLLECTION=knowledge_base

# Automatically index chapters after generation
QMD_AUTO_INDEX=true

# Index intermediate drafts (not just final chapters)
QMD_INDEX_DRAFTS=false

# Minimum relevance score for search results (0.0-1.0)
QMD_MIN_SCORE=0.3

# Maximum search results to return
QMD_MAX_RESULTS=5
"""


def create_full_env_template(filepath: str = ".env.example") -> None:
    """Create a complete .env file template with all options including QMD"""
    full_template = ENV_TEMPLATE.strip() + QMD_ENV_TEMPLATE
    with open(filepath, "w") as f:
        f.write(full_template)
