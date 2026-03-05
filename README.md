# AutoGen Book Generator

A Python-based system that uses AutoGen to generate complete books through collaborative AI agents. The system employs multiple specialized agents working together to create coherent, structured narratives from initial prompts.

## ✅ Improvements Made

This codebase has been significantly improved with the following enhancements:

### Code Quality
- ✅ **Custom Exception Hierarchy** - Specific exceptions for different failure modes (LLMError, ParseError, ValidationError, etc.)
- ✅ **Type Safety** - Added proper type hints with return annotations throughout
- ✅ **Pydantic Models** - Data validation with Chapter, Outline, and other models
- ✅ **Logging Module** - Replaced print statements with Python's logging module
- ✅ **Constants Module** - All magic numbers extracted to constants
- ✅ **Pre-compiled Regex** - All regex patterns compiled for performance

### Configuration
- ✅ **Environment Variables** - Support via python-dotenv (.env.example included)
- ✅ **Multiple LLM Providers** - Support for local, OpenAI, and Azure
- ✅ **Dataclass-based Config** - Typed configuration with validation
- ✅ **Configuration Validation** - Input validation with helpful error messages

### Error Handling & Robustness
- ✅ **Retry Logic** - Exponential backoff with jitter for LLM calls
- ✅ **Graceful Degradation** - Emergency processing for incomplete outlines
- ✅ **Specific Exception Handling** - Different handling for different error types
- ✅ **Content Validation** - Word count, chapter sequence integrity checks

### Data Validation
- ✅ **Schema Validation** - Pydantic models for outline and chapter data
- ✅ **Chapter Sequence Verification** - Ensures all chapters are present and in order
- ✅ **Content Length Validation** - Word count validation with configurable minimums
- ✅ **Input Sanitization** - Filename sanitization and content cleaning

### Testing & Quality
- ✅ **Test Suite** - pytest configuration and initial tests
- ✅ **Type Checking Config** - mypy support

## Features

- **AutoGen 2.0 Support** - Full support for AutoGen 2.0 (autogen-agentchat) with async execution
- **Backward Compatibility** - Falls back to AutoGen 0.2 (pyautogen) when needed
- Multi-agent collaborative writing system
- Structured chapter generation with consistent formatting
- Maintains story continuity and character development
- Automated world-building and setting management
- Support for complex, multi-chapter narratives
- Built-in validation and error handling
- Retry logic with exponential backoff
- Multiple LLM provider support
- Comprehensive logging

## Architecture

The system uses several specialized agents:

- **Story Planner**: Creates high-level story arcs and plot points
- **World Builder**: Establishes and maintains consistent settings
- **Memory Keeper**: Tracks continuity and context
- **Writer**: Generates the actual prose
- **Editor**: Reviews and improves content
- **Outline Creator**: Creates detailed chapter outlines

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autogen-book-generator.git
cd autogen-book-generator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

**For AutoGen 2.0 (recommended):**
```bash
pip install -r requirements.txt
# This installs: autogen-agentchat>=0.4.0 and autogen-ext[openai]>=0.4.0
```

**For legacy AutoGen 0.2:**
```bash
pip install pyautogen>=0.2.0
pip install -r requirements.txt  # Install other dependencies
```

4. Create environment configuration (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

## Runtime Instructions

### Quick Start - Run with Default Settings

The fastest way to start generating a book:

```bash
python main.py
```

This will:
1. Load configuration from `.env` file (or use defaults)
2. Generate a 25-chapter book outline based on the default story prompt
3. Generate all 25 chapters sequentially
4. Save output to `book_output/` directory

**Expected Runtime**: 30-60 minutes depending on your LLM speed and hardware

### Running with Custom Configuration

```bash
# Set environment variables inline
BOOK_NUM_CHAPTERS=10 BOOK_MIN_WORDS=3000 python main.py

# Or use a specific provider
LLM_PROVIDER=openai OPENAI_API_KEY=sk-xxx python main.py
```

### Loading Story from Markdown File

You can write your story outline in a markdown file and pass it to the generator:

```bash
# Load story from a markdown file
python main.py --prompt my_story.md

# Or use the short option
python main.py -p my_story.md
```

Example `my_story.md`:
```markdown
# My Science Fiction Novel

## Premise
In the year 2150, humanity discovers an ancient alien artifact on Mars...

## Characters
- **Dr. Sarah Chen**: Lead archaeologist, brilliant but impulsive
- **Commander Rodriguez**: Cautious military leader

## Key Themes
- First contact with alien intelligence
- The ethics of technological advancement
- Human survival against cosmic odds
```

The system will read the entire file content and use it as the story prompt.

### Command-Line Options

```bash
# Load from file with specific chapter count
python main.py --prompt story.md --chapters 10

# Use specific provider
python main.py --prompt story.md --provider openai

# Custom output directory
python main.py --prompt story.md --output ./my_book

# Set log level for debugging
python main.py --prompt story.md --log-level DEBUG

# Combine options
python main.py -p story.md -c 15 --provider local -o ./output

# Force AutoGen 2.0 mode (requires AutoGen 2.0 packages)
python main.py --autogen2

# Force legacy AutoGen mode
python main.py --legacy-autogen

# Show all available options
python main.py --help
```

### Runtime Behavior

When you run the application, you'll see output like:

```
==================================================
Starting Book Generator
==================================================
2024-01-15 10:30:45 - book_generator.main - INFO - Using LLM provider: local
2024-01-15 10:30:45 - book_generator.main - INFO - Output directory: book_output
2024-01-15 10:30:45 - book_generator.main - INFO - Number of chapters: 25

==================================================
Phase 1: Generating Book Outline
==================================================
2024-01-15 10:30:46 - outline_generator - INFO - OutlineGenerator initialized
2024-01-15 10:30:46 - book_generator.main - INFO -
Generated Outline (25 chapters):
2024-01-15 10:30:46 - book_generator.main - INFO -
Chapter 1: The Beginning
--------------------------------------------------
2024-01-15 10:30:46 - book_generator.main - INFO - Chapter 2: Rising Tension
...

==================== Chapter 1 ====================
2024-01-15 10:31:00 - book_generator - INFO - Generating Chapter 1...
2024-01-15 10:32:30 - book_generator - INFO - ✓ Saved to: book_output/chapter_01.txt
✓ Chapter 1 complete

==================== Chapter 2 ====================
...
```

### Progress Monitoring

The application provides real-time feedback:
- **Phase indicators**: Shows which phase (outline vs chapters) is running
- **Chapter progress**: Displays current chapter being generated
- **Completion marks**: ✓ indicates successful chapter completion
- **Error handling**: Clear error messages if something goes wrong

### Expected Output

After successful completion, you'll have:

```
book_output/
├── outline.txt              # The complete book outline
├── chapter_01.txt           # ~3000+ words each
├── chapter_02.txt
├── chapter_03.txt
├── ...
└── chapter_25.txt
```

### Stopping and Resuming

**To stop**: Press `Ctrl+C` - the application will gracefully exit at the next checkpoint

**To resume**: Simply run again - the system will:
1. Check which chapters already exist
2. Skip completed chapters
3. Continue from where it left off

### Dual Model Runtime

If using two models (creative + planning), the runtime output shows:

```bash
$ LOCAL_CREATIVE_MODEL=Mistral-7B LOCAL_PLANNING_MODEL=Mixtral-8x7B python main.py
...
2024-01-15 10:30:45 - agents - INFO - Created writer agent (using creative model: Mistral-7B)
2024-01-15 10:30:45 - agents - INFO - Created editor agent (using planning model: Mixtral-8x7B)
```

### Troubleshooting Runtime Issues

**If generation is slow**:
- Check LLM server logs for errors
- Reduce `BOOK_NUM_CHAPTERS` in `.env`
- Reduce `BOOK_MIN_WORDS` requirement

**If chapters are too short**:
- The system will retry automatically (max 3 attempts)
- Check logs for "ChapterTooShortError"
- Increase timeout in `.env`: `DEFAULT_TIMEOUT=1200`

**If outline generation fails**:
- Check LLM connection: `curl http://localhost:1234/v1/models`
- Try emergency processing mode (automatic fallback)
- Check logs for specific parsing errors

## Configuration

The system can be configured through:

1. **Environment Variables** (in `.env` file):
```bash
# LLM Provider Selection
LLM_PROVIDER=local  # Options: local, openai, azure

# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4

# Azure Configuration
AZURE_API_KEY=your-azure-key-here
AZURE_DEPLOYMENT=your-deployment-name
AZURE_BASE_URL=https://your-resource.openai.azure.com

# Book Generation Settings
BOOK_NUM_CHAPTERS=25
BOOK_MIN_WORDS=3000
BOOK_OUTPUT_DIR=book_output

# Logging
BOOK_LOG_LEVEL=INFO
```

2. **Programmatic Configuration**:
```python
from config import get_app_config

config = get_app_config(
    provider="openai",  # or "local", "azure"
    output_dir="my_book"
)

# Validate configuration
config.validate()

# Get agent configuration
agent_config = config.get_agent_config()
```

## Usage

### Basic Usage

```python
from main import run_book_generation
from config import get_app_config

# Using environment variables
run_book_generation()

# Or with explicit configuration
config = get_app_config(provider="openai")
run_book_generation(config=config)
```

### Custom Story Prompt

```python
from main import run_book_generation

my_prompt = """
Write a science fiction story about...
"""

run_book_generation(custom_prompt=my_prompt)
```

### Advanced Usage

```python
from config import get_app_config, AppConfig
from agents import BookAgents
from book_generator import BookGenerator
from outline_generator import OutlineGenerator

# Get configuration
config = get_app_config()
agent_config = config.get_agent_config()

# Create agents
outline_agents = BookAgents(agent_config.to_dict(), num_chapters=25)
agents = outline_agents.create_agents(initial_prompt, num_chapters=25)

# Generate outline
outline_gen = OutlineGenerator(agents, agent_config.to_dict())
outline = outline_gen.generate_outline(initial_prompt, num_chapters=25)

# Initialize book generator
book_agents = BookAgents(agent_config.to_dict(), outline=outline, num_chapters=25)
agents_with_context = book_agents.create_agents(initial_prompt, num_chapters=25)
book_gen = BookGenerator(
    agents_with_context, 
    agent_config.to_dict(), 
    outline,
    output_dir=config.output_dir
)

# Generate book
book_gen.generate_book(outline)

# Get statistics
stats = book_gen.get_book_stats()
print(f"Total words: {stats['total_words']}")
```

## Output Structure

Generated content is saved in the `book_output` directory:
```
book_output/
├── outline.txt
├── chapter_01.txt
├── chapter_02.txt
├── ...
└── chapter_25.txt
```

Chapter files include:
- Chapter number in header
- Full chapter content
- Word count validation (minimum 3000 words by default)

## Error Handling

The system includes robust error handling:

### Specific Exceptions

- `LLMError` - LLM API call failures
- `LLMTimeoutError` - Request timeouts
- `LLMRateLimitError` - Rate limiting
- `ParseError` - Content parsing failures
- `ValidationError` - Data validation failures
- `ChapterError` - Chapter generation failures
- `ChapterTooShortError` - Content too short
- `ChapterIncompleteError` - Incomplete generation sequence
- `ConfigurationError` - Invalid configuration
- `FileOperationError` - File I/O failures

### Retry Behavior

The system automatically retries failed operations with exponential backoff:
- Base delay: 1 second
- Exponential factor: 2.0
- Max delay: 60 seconds
- Max retries: 3
- Jitter: 10% randomization

## Requirements

- Python 3.8+
- **AutoGen 2.0 (recommended)**: `autogen-agentchat>=0.4.0` and `autogen-ext[openai]>=0.4.0`
- **Or AutoGen 0.2 (legacy)**: `pyautogen>=0.2.0`
- pydantic>=2.0.0
- python-dotenv>=1.0.0
- Other dependencies in requirements.txt

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black *.py
```

### Type Checking

```bash
mypy *.py
```

### Configuration Template

Generate a `.env` template:

```python
from config import create_env_template
create_env_template(".env")
```

## Project Structure

```
.
├── agents.py              # Agent configuration and creation
├── book_generator.py      # Chapter generation logic
├── config.py              # Configuration management
├── constants.py           # All constants and regex patterns
├── exceptions.py          # Custom exception classes
├── main.py                # Entry point and orchestration
├── models.py              # Pydantic data models
├── outline_generator.py   # Outline generation logic
├── utils.py               # Utility functions and helpers
├── test_book_generator.py # Unit tests
├── requirements.txt       # Dependencies
├── .env.example          # Environment configuration template
└── pytest.ini            # Test configuration
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using the [AutoGen](https://github.com/microsoft/autogen) framework
- Inspired by collaborative writing systems