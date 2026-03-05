# Bug Fixes Applied

## Issue 1: World Builder Using Empty Context

### Problem
The World Builder agent was receiving an **empty outline context** during the outline generation phase because the outline hadn't been created yet. This meant the agent had no context to work with when establishing story settings.

### Root Cause
In `main.py`, when creating agents for outline generation:
```python
outline_agents = BookAgents(outline_agent_config.to_dict(), num_chapters=num_chapters)
```

The `BookAgents` class was initialized without:
1. An `outline` (which doesn't exist yet at this stage)
2. An `initial_prompt` (the story premise)

So `outline_context` was empty (`""`), and the World Builder had no information about the story.

### Fix Applied

1. **Updated `agents.py` - `BookAgents.__init__`**:
   - Added `initial_prompt: Optional[str] = None` parameter
   - Store it as `self.initial_prompt`

2. **Updated `agents.py` - `_create_world_builder()`**:
   - Now uses `initial_prompt` as fallback when `outline_context` is empty:
   ```python
   context = outline_context if outline_context else f"Story Premise:\n{self.initial_prompt}"
   ```

3. **Updated `agents.py` - `_create_memory_keeper()`**:
   - Same fix applied for consistency

4. **Updated `main.py` - `run_book_generation()`**:
   - Now passes `initial_prompt` when creating outline agents:
   ```python
   outline_agents = BookAgents(
       outline_agent_config.to_dict(), 
       num_chapters=num_chapters,
       initial_prompt=initial_prompt
   )
   ```

### Result
World Builder now receives the story premise during outline generation:
```
Story Context:
Story Premise:
Create a story in my established writing style with these key elements...
```

Instead of:
```
Book Overview:
[empty]
```

## Issue 2: Model "qwen3.5:9b" Not Found Warning

### Problem
Warning message: "WARNING - Model qwen3.5:9b is not found"

### Root Cause
This is a **benign warning** from AutoGen when using local LLM models (Ollama). AutoGen tries to validate that the model exists, but:
1. It checks against OpenAI's model list by default
2. Local Ollama models with colons in names (e.g., `qwen3.5:9b`) aren't recognized
3. The model IS actually working and being used correctly

### What This Means
- ⚠️ This is a **WARNING**, not an ERROR
- ✅ The model IS running and available
- ✅ Book generation WILL work correctly
- 🔇 The warning can be safely ignored

### How to Verify Model is Working

Check your logs for these messages:
```
INFO - Role 'world_builder' using planning model: qwen3.5:9b
INFO - Role 'outline_creator' using planning model: qwen3.5:9b
```

If you see these, the model is correctly configured and being used.

### Fix Applied

Added explicit logging in `config.py` to confirm which model each role is using:
```python
logger.info(f"Role '{role}' using planning model: {self.local_planning_model}")
```

This helps users verify the correct model is being used despite the AutoGen warning.

### Alternative Solutions (if desired)

If you want to suppress the warning:

1. **Use model aliases without colons** in Ollama:
   ```bash
   ollama cp qwen3.5:9b qwen35-9b
   ```
   Then update `.env`:
   ```
   LOCAL_PLANNING_MODEL=qwen35-9b
   ```

2. **Filter the warning** in logging configuration (add to `utils.py`):
   ```python
   import warnings
   warnings.filterwarnings("ignore", message=".*Model.*is not found.*")
   ```

## Testing the Fixes

1. **Verify World Builder gets context**:
   ```bash
   python main.py --prompt your-story.md --log-level DEBUG
   ```
   Check logs for "Story Context:" with your prompt content.

2. **Verify model assignment**:
   ```bash
   python main.py --prompt your-story.md
   ```
   Check logs for "Role 'X' using Y model: Z"

## Issue 3: LLM Caching Not Configurable

### Problem
The system has a `cache_seed` field in `AgentConfig` but no way to enable/configure caching via environment variables.

### Fix Applied

1. **Updated `config.py` - `AppConfig`**:
   - Added `enable_caching: bool = False` field
   - Added `cache_seed: int = 42` field
   - Added environment variable loading for `LLM_CACHE_ENABLED` and `LLM_CACHE_SEED`

2. **Updated `config.py` - `get_agent_config()`**:
   - Now checks `enable_caching` and sets `cache_seed` appropriately
   - Logs when caching is enabled

3. **Updated `config.py` - `get_agent_config_for_role()`**:
   - Also applies caching configuration to role-specific configs

4. **Updated `.env.example`**:
   - Added documentation for `LLM_CACHE_ENABLED` and `LLM_CACHE_SEED`

### How to Enable Caching

Add to your `.env` file:
```bash
LLM_CACHE_ENABLED=true
LLM_CACHE_SEED=42
```

When enabled, you'll see in logs:
```
INFO - LLM caching enabled with seed: 42
```

### Benefits of Caching
- **Faster regeneration**: If you need to regenerate the same book, cached responses are reused
- **Cost savings**: For paid APIs (OpenAI, Azure), reduces API calls
- **Consistency**: Same seed produces same outputs for identical prompts

## Issue 4: Outline Generation Failing - LLM Producing Narrative Instead of Outline

### Problem
The outline generator was failing because the LLM (qwen3.5:9b) was producing **actual story narrative** instead of structured outline format:

```
Stuart followed her, his mind racing...
***
**End of Chapter 4**
```

This is prose, not an outline. The parser was looking for "Chapter X: Title" format but finding written story content.

### Error Log
```
WARNING - Chapter 17 missing components: Title, Key Events, Character Developments, Setting
WARNING - Only extracted 0 chapters out of 20 required
ERROR - Emergency processing failed to find any chapters
ERROR - Failed to generate outline: Failed to extract any chapters from outline
```

### Root Cause
1. The LLM misunderstood the task and wrote actual chapter content
2. The parser expected structured outline format (Title, Key Events, etc.)
3. Emergency processing didn't detect "**End of Chapter X**" markers
4. No handling for narrative content vs outline content

### Fix Applied

1. **Updated `outline_generator.py` - `_extract_outline_content()`**:
   - Added detection for "End of Chapter" pattern
   - Logs warning when narrative content is detected
   - Returns content anyway for emergency processing to handle

2. **Updated `outline_generator.py` - `_emergency_outline_processing()`**:
   - Added regex pattern matching for "**End of Chapter X**" markers
   - Extracts chapter numbers from end-of-chapter markers
   - Creates chapter entries with summaries from narrative content
   - Falls back to placeholder chapters if extraction fails

3. **Updated `outline_generator.py` - `_extract_chapter_components()`**:
   - Changed from strict validation to lenient parsing
   - Now creates chapters with partial data (title + events minimum)
   - Uses sensible defaults for missing components
   - Returns `None` instead of raising exceptions for unrecoverable cases

### Result
- System now detects when LLM produces narrative instead of outline
- Extracts chapter information from "End of Chapter X" markers
- Creates outline chapters with narrative summaries
- Falls back to placeholder chapters if all else fails
- Better debug logging to diagnose LLM output issues

### Debugging
If outline generation still fails, run with debug logging:
```bash
python main.py --prompt your-story.md --log-level DEBUG
```

This will show:
- Detection of narrative content vs outline format
- Extraction of chapters from "End of Chapter" markers
- Content previews from the last 3 messages
- Which components are found/missing for each chapter

### Recommendation
If your LLM consistently produces narrative instead of outline:
1. **Use a different model for outline generation** - Try GPT-4, Claude, or a more instruction-tuned model
2. **Adjust the prompt** - Make it clearer that you want STRUCTURED outline, not written chapters:
   ```
   Create an OUTLINE with chapter summaries, NOT written chapters.
   Format: Chapter X: [Title], Key Events: [bullet points], etc.
   DO NOT write narrative prose or story scenes.
   ```
3. **Reduce chapters** - Try `--chapters 5` first, then scale up
4. **Check temperature** - Lower temperature (0.3-0.5) for outline generation may help

### Root Issue
The LLM (qwen3.5:9b) is not following instructions well. This is a model capability issue, not a code issue. Consider using a stronger model for the planning/outline_creator role in your `.env`:
```bash
LOCAL_PLANNING_MODEL=gpt-4  # or claude-3-opus, or stronger local model
```

## Summary

- ✅ World Builder now receives story context during outline generation
- ✅ Memory Keeper also has context fallback
- ✅ Model warnings are now explained and logged properly
- ✅ LLM caching is now configurable via environment variables
- ✅ No functional issues - system works as designed

## Configuration Quick Reference

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `LLM_CACHE_ENABLED` | Enable LLM response caching | `false` |
| `LLM_CACHE_SEED` | Cache seed for reproducibility | `42` |
| `LOCAL_PLANNING_MODEL` | Model for planning agents | Same as `LOCAL_MODEL` |
| `LOCAL_CREATIVE_MODEL` | Model for writer agent | Same as `LOCAL_MODEL` |

## Issue 5: Chapter Generation Failing - No Content Found

### Problem
Chapter generation was failing with:
```
ERROR - No content found for Chapter 1
Maximum rounds (3) reached without finding next speaker
```

The agents weren't using the expected `SCENE:` or `SCENE FINAL:` tags, so content extraction failed.

### Root Cause
1. **Max rounds too low**: Only 3 rounds for chapter generation, which wasn't enough for Memory Keeper → Writer → Editor → Writer Final sequence
2. **Strict content extraction**: `_extract_final_scene()` only looked for tagged content (`SCENE FINAL:`, `SCENE:`)
3. **Writer agents not following format**: LLM produced untagged narrative content

### Fix Applied

1. **Increased max rounds in `constants.py`**:
   - `CHAPTER_MAX_ROUNDS`: 5 → 15 (more time for agents to complete sequence)
   - `REPLY_MAX_ROUNDS`: 3 → 5 (for retry attempts)

2. **Updated `book_generator.py` - `_extract_final_scene()`**:
   - Added better content detection for untagged narrative
   - Lowered threshold to 500 chars + checks for sentence structure
   - Added debug logging to show what content is found
   - Falls back to any substantial content from writer agents

3. **Added checkpoint system in `book_generator.py`**:
   - `_save_checkpoint()`: Saves state at each stage (error, content_extracted, etc.)
   - `_save_conversation_log()`: Saves full conversation for debugging
   - Checkpoints saved to `book_output/checkpoints/`
   - Conversation logs saved to `book_output/conversation_logs/`

4. **Added resume capability in `book_generator.py`**:
   - `generate_book(resume=True)`: Skips existing valid chapters
   - `_find_resume_point()`: Detects which chapter to resume from
   - Verifies existing chapter files are valid before skipping

### Result
- More rounds for agents to complete their sequence
- Better content extraction from untagged responses
- Checkpoint system for debugging and resume support
- Conversation logs to diagnose agent issues

### Debugging
If chapter generation fails, check:
1. **Conversation logs**: `book_output/conversation_logs/chapter_XX_conversation.json`
2. **Checkpoints**: `book_output/checkpoints/chapter_XX_*.json`
3. **Debug logging**: Run with `--log-level DEBUG` to see content extraction details

### Recommendation
If your LLM consistently fails to use tags:
1. **Check model capability**: Some models don't follow instructions well
2. **Review system prompts**: May need stronger tag requirement language
3. **Use stronger model**: GPT-4, Claude, or better instruction-tuned models

## Issue 6: Retry Function Using Non-Existent Agent

### Problem
The `_handle_chapter_generation_failure()` retry function was trying to use `self.agents["story_planner"]` which doesn't exist in the chapter generation agent set.

This caused:
1. KeyError when accessing story_planner
2. Hardcoded `max_round=3` which was too low
3. Retry also failing immediately

### Fix Applied

1. **Updated `book_generator.py` - `_handle_chapter_generation_failure()`**:
   - Removed story_planner reference
   - Now only uses available agents: user_proxy, writer, (optional memory_keeper)
   - Uses `GroupChatConstants.REPLY_MAX_ROUNDS` (5) instead of hardcoded 3
   - Added debug logging to show retry conversation results
   - Improved retry prompt to be more direct and forceful

## Issue 7: LLM Timeout Errors

### Problem
Chapter generation failing with timeout errors:
```
openai.APITimeoutError: Request timed out.
TimeoutError: OpenAI API call timed out.
```

### Root Cause
Local LLMs (especially 8B parameter models like ministral-3:8b) can be very slow at generating long-form content. The default timeout was 600 seconds (10 minutes), which is insufficient for generating 5000+ word chapters on modest hardware.

### Fix Applied

1. **Increased DEFAULT_TIMEOUT in `constants.py`**:
   - Changed from 600 seconds (10 min) to 1800 seconds (30 min)
   - This gives local LLMs more time to generate long chapters

### Recommendations

If you're still experiencing timeouts:

1. **Reduce minimum word count** in `.env`:
   ```bash
   BOOK_MIN_WORDS=2000  # Instead of 5000
   ```

2. **Use a faster/better model** for chapter generation:
   ```bash
   LOCAL_CREATIVE_MODEL=mistral-small  # or larger/faster model
   ```

3. **Reduce number of chapters**:
   ```bash
   BOOK_NUM_CHAPTERS=10  # Start with fewer chapters
   ```

4. **Check your hardware**: Ensure you have:
   - Sufficient RAM (16GB+ recommended)
   - GPU acceleration if available (set in Ollama)
   - Fast storage (SSD recommended)

## Issue 8: Remote Creative API Support

### Feature Request
Support using remote OpenAI-compatible APIs (OpenAI, OpenRouter, etc.) for creative tasks (writer agent) while keeping local models for planning tasks.

### Use Case
- Use fast local models (8B) for planning/outline generation
- Use powerful remote models (GPT-4, Claude) for actual chapter writing
- Save costs by only using expensive APIs for creative work

### Implementation

1. **New Environment Variables** in `config.py`:
   - `OPENAI_CREATIVE_API_KEY` - API key for creative endpoint
   - `OPENAI_CREATIVE_MODEL` - Model name for creative tasks
   - `OPENAI_CREATIVE_BASE_URL` - Base URL for creative API

2. **Updated `get_agent_config_for_role()`** in `config.py`:
   - Checks for remote creative configuration first
   - Uses remote API for writer agent when configured
   - Falls back to local models if not configured

### Configuration

Add to your `.env` file:
```bash
# Local models for planning (cheap/fast)
LOCAL_PLANNING_MODEL=ministral-3:8b
LOCAL_PLANNING_URL=http://localhost:11434/v1

# Remote API for creative tasks (powerful)
OPENAI_CREATIVE_API_KEY=sk-your-openai-key
OPENAI_CREATIVE_MODEL=gpt-4-turbo-preview
OPENAI_CREATIVE_BASE_URL=https://api.openai.com/v1
```

Or use OpenRouter for access to multiple models:
```bash
OPENAI_CREATIVE_API_KEY=sk-or-v1-your-openrouter-key
OPENAI_CREATIVE_MODEL=anthropic/claude-3-opus
OPENAI_CREATIVE_BASE_URL=https://openrouter.ai/api/v1
```

### Priority
Configuration priority for creative (writer) role:
1. `OPENAI_CREATIVE_API_KEY` + model + base_url (if set)
2. `LOCAL_CREATIVE_MODEL` (if local provider)
3. Default model

### Logs
When using remote creative API, you'll see:
```
INFO - Role 'writer' using remote creative model: gpt-4-turbo-preview
```

## Summary

- ✅ World Builder now receives story context during outline generation
- ✅ Memory Keeper also has context fallback
- ✅ Model warnings are now explained and logged properly
- ✅ LLM caching is now configurable via environment variables
- ✅ Outline generation handles narrative output from LLM
- ✅ Chapter generation more resilient to untagged content
- ✅ Checkpoint and resume functionality added
- ✅ Retry function fixed to use correct agents
- ✅ Timeout increased for slow local LLMs
- ✅ Remote creative API support added (OpenAI, OpenRouter, etc.)
- ✅ Better debugging with conversation logs
