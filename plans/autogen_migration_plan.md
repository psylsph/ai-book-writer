# AutoGen Migration Plan: Legacy → Latest Version

## Executive Summary

This document outlines the migration strategy from **AutoGen Legacy API** (pyautogen<0.2.0) to **AutoGen 2.0** (autogen-agentchat/pyautogen>=0.2.0).

**Current State**: Codebase uses legacy AutoGen API with `ConversableAgent`, `GroupChat`, and `GroupChatManager`

**Target State**: Migrate to AutoGen 2.0 with new agent, team, and runner APIs

**Estimated Effort**: Significant refactoring across 4 core files

---

## Key API Changes

### 1. Agent Creation

#### Legacy API (Current)
```python
import autogen

agent = autogen.ConversableAgent(
    name="writer",
    system_message="You are a writer...",
    llm_config={
        "config_list": [{"model": "gpt-4", "api_key": "..."}],
        "temperature": 0.7
    }
)
```

#### New API (Target)
```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model="gpt-4",
    api_key="..."
)

agent = AssistantAgent(
    name="writer",
    system_message="You are a writer...",
    model_client=model_client,
    temperature=0.7
)
```

### 2. Group Chat Management

#### Legacy API (Current)
```python
groupchat = autogen.GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=10,
    speaker_selection_method="round_robin"
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

result = agent1.initiate_chat(manager, message="Hello")
```

#### New API (Target)
```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

team = RoundRobinGroupChat(
    participants=[agent1, agent2, agent3],
    max_turns=10
)

# Run the team
result = await team.run(task="Hello")
# Or with streaming:
await team.run_stream(task="Hello", Console())
```

### 3. Message Handling

#### Legacy API (Current)
```python
# Messages stored in groupchat.messages
for msg in groupchat.messages:
    content = msg.get("content")
    sender = msg.get("name")
```

#### New API (Target)
```python
# Messages from team.run() result
for msg in result.messages:
    content = msg.content
    sender = msg.source  # agent name
```

---

## File-by-File Migration Strategy

### 1. agents.py

**Changes Required:**
- Replace `autogen.ConversableAgent` with `AssistantAgent` or `CodingAssistantAgent`
- Update agent configuration from `llm_config` dict to `model_client` object
- Update all agent creation methods

**Example Migration:**

```python
# BEFORE
def _create_writer(self) -> autogen.ConversableAgent:
    return autogen.ConversableAgent(
        name="writer",
        system_message="You are a creative writer...",
        llm_config=self.agent_config,
    )

# AFTER
def _create_writer(self) -> AssistantAgent:
    model_client = self._create_model_client("writer")
    return AssistantAgent(
        name="writer",
        system_message="You are a creative writer...",
        model_client=model_client,
    )
```

### 2. agent_factory.py

**Changes Required:**
- Replace `autogen.GroupChat` with `RoundRobinGroupChat` or custom team
- Remove `GroupChatManager` usage
- Update all factory methods

**Example Migration:**

```python
# BEFORE
def create_outline_chat(agents, max_rounds=4) -> autogen.GroupChat:
    return autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=max_rounds,
        speaker_selection_method="round_robin"
    )

# AFTER
def create_outline_chat(agents, max_turns=4) -> RoundRobinGroupChat:
    return RoundRobinGroupChat(
        participants=agents,
        max_turns=max_turns
    )
```

### 3. book_generator.py

**Changes Required:**
- Replace `_create_group_chat()` to create teams instead of GroupChat
- Replace `initiate_chat()` with `team.run()` or `team.run_stream()`
- Update message extraction logic
- Update retry logic

**Example Migration:**

```python
# BEFORE
def generate_chapter(self, chapter_number: int, prompt: str) -> None:
    groupchat = self._create_group_chat()
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=self.agent_config
    )
    
    self.agents["user_proxy"].initiate_chat(
        manager,
        message=chapter_prompt
    )
    
    messages = groupchat.messages
    self._process_chapter_results(chapter_number, messages)

# AFTER
async def generate_chapter(self, chapter_number: int, prompt: str) -> None:
    team = self._create_team()
    
    result = await team.run(task=chapter_prompt)
    
    messages = result.messages
    self._process_chapter_results(chapter_number, messages)
```

### 4. outline_generator.py

**Changes Required:**
- Similar changes to book_generator.py
- Replace GroupChatManager with team.run()
- Update result processing

---

## Configuration Changes

### config.py

**New Model Client Creation:**

```python
from autogen_ext.models import OpenAIChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

def create_model_client(self, role: str):
    """Create a model client for the given role"""
    if self.provider == "local":
        from autogen_ext.models import OpenAIChatCompletionClient
        return OpenAIChatCompletionClient(
            model=self.local_model,
            base_url=self.local_url,
            api_key=self.local_api_key
        )
    elif self.provider == "openai":
        return OpenAIChatCompletionClient(
            model=self.openai_model,
            api_key=self.openai_api_key
        )
    # ... other providers
```

---

## Breaking Changes to Handle

1. **Async/Await**: New API uses async methods extensively
   - All `team.run()` calls become `await team.run()`
   - Need to update function signatures to `async def`

2. **Message Format**: Message objects are different
   - Old: `{"content": "...", "role": "...", "name": "..."}`
   - New: `Message(content="...", source="...", ...)`

3. **No GroupChatManager**: Replaced by Team/Runner pattern
   - Old: Create GroupChat, pass to GroupChatManager
   - New: Create Team directly with participants

4. **Speaker Selection**: Built into team types
   - `RoundRobinGroupChat` for round-robin
   - `SelectorGroupChat` for LLM-based selection
   - Custom teams for complex logic

---

## Testing Strategy

### Phase 1: Unit Tests
```python
# Test individual agent creation
async def test_agent_creation():
    agent = create_writer_agent()
    assert agent.name == "writer"
    assert isinstance(agent, AssistantAgent)

# Test team creation
async def test_team_creation():
    team = create_outline_team()
    assert len(team.participants) == 4
```

### Phase 2: Integration Tests
```python
# Test outline generation
async def test_outline_generation():
    generator = OutlineGenerator(agents)
    outline = await generator.generate_outline(premise)
    assert len(outline) > 0

# Test chapter generation
async def test_chapter_generation():
    generator = BookGenerator(agents)
    await generator.generate_chapter(1, prompt)
    assert os.path.exists("chapter_01.md")
```

### Phase 3: End-to-End Tests
```python
# Test full book generation
async def test_book_generation():
    agents = create_all_agents()
    outline = await generate_outline(agents, premise)
    book = await generate_book(agents, outline)
    assert len(book.chapters) == len(outline)
```

---

## Rollback Strategy

If migration fails, we can rollback by:

1. **Git Revert**: Revert all migration commits
2. **Reinstall Legacy**: `pip install 'pyautogen<0.2.0'`
3. **Restore Tests**: Ensure all tests pass with legacy API

**Recommendation**: Create a feature branch `autogen-2.0-migration` for this work.

---

## Dependencies Update

### requirements.txt

```txt
# Core dependencies
# AutoGen 2.0 - new architecture
autogen-agentchat>=0.4.0  # Core agent and team APIs
autogen-ext[openai]>=0.4.0  # Extensions for OpenAI, Azure, etc.
pydantic>=2.0.0
python-dotenv>=1.0.0
```

---

## Migration Checklist

### Pre-Migration
- [ ] Create feature branch
- [ ] Backup current working state
- [ ] Run all tests to establish baseline
- [ ] Document current behavior

### Migration
- [ ] Update requirements.txt
- [ ] Migrate config.py (model client creation)
- [ ] Migrate agents.py (agent creation)
- [ ] Migrate agent_factory.py (team creation)
- [ ] Migrate book_generator.py (team usage)
- [ ] Migrate outline_generator.py (team usage)
- [ ] Update all type hints
- [ ] Update imports

### Post-Migration
- [ ] Fix all test failures
- [ ] Run integration tests
- [ ] Run end-to-end tests
- [ ] Performance testing
- [ ] Update documentation
- [ ] Create migration guide
- [ ] Code review
- [ ] Merge to main

---

## Resources

- [AutoGen Migration Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html)
- [AutoGen 2.0 Documentation](https://microsoft.github.io/autogen/stable/)
- [AutoGen GitHub Repository](https://github.com/microsoft/autogen)

---

## Questions & Considerations

1. **Async Migration**: Should we make the entire codebase async, or create async wrappers?
2. **Backward Compatibility**: Do we need to support both APIs during transition?
3. **Performance**: Will the new API perform differently? Need benchmarks.
4. **Error Handling**: New error patterns may require updated exception handling.
5. **Testing Coverage**: Do we have sufficient tests to validate migration?

---

## Next Steps

1. Review this plan with the team
2. Create feature branch
3. Start with Phase 1 (config.py migration)
4. Test incrementally after each file migration
5. Document any deviations from this plan

**Last Updated**: 2026-03-05
**Status**: Planning Phase - Ready for Implementation
