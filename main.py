"""Main script for running the book generation system"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from agents import BookAgents
from book_generator import BookGenerator
from config import get_app_config, AppConfig, AUTOGEN_2_AVAILABLE
from outline_generator import OutlineGenerator
from utils import get_logger, setup_logging, save_outline_to_file

# Determine whether to use AutoGen 2.0 by default
USE_AUTOGEN2_DEFAULT = AUTOGEN_2_AVAILABLE


def load_prompt_from_file(filepath: str) -> str:
    """Load story prompt from a markdown or text file

    Args:
        filepath: Path to the file containing the story outline

    Returns:
        The file contents as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    content = path.read_text(encoding='utf-8').strip()

    if not content:
        raise ValueError(f"Prompt file is empty: {filepath}")

    logger = get_logger("main")
    logger.info(f"Loaded prompt from {filepath} ({len(content)} characters)")

    return content


def get_initial_prompt() -> str:
    """Get the initial story prompt"""
    return """
    Create a story in my established writing style with these key elements:
    Its important that it has several key storylines that intersect and influence each other. The story should be set in a modern corporate environment, with a focus on technology and finance. The protagonist is a software engineer named Dane who has just completed a groundbreaking stock prediction algorithm. The algorithm predicts a catastrophic market crash, but Dane oversleeps and must rush to an important presentation to share his findings with executives. The tension arises from the questioning of whether his "error" might actually be correct.

    The piece is written in third-person limited perspective, following Dane's thoughts and experiences. The prose is direct and technical when describing the protagonist's work, but becomes more introspective during personal moments. The author employs a mix of dialogue and internal monologue, with particular attention to time progression and technical details around the algorithm and stock predictions.
    Story Arch:

    Setup: Dane completes a groundbreaking stock prediction algorithm late at night
    Initial Conflict: The algorithm predicts a catastrophic market crash
    Rising Action: Dane oversleeps and must rush to an important presentation
    Climax: The presentation to executives where he must explain his findings
    Tension Point: The questioning of whether his "error" might actually be correct

    Characters:

    Dane: The protagonist; a dedicated software engineer who prioritizes work over personal life. Wears grey polo shirts on Thursdays, tends to get lost in his work, and struggles with work-life balance. More comfortable with code than public speaking.
    Gary: Dane's nervous boss who seems caught between supporting Dane and managing upper management's expectations
    Jonathan Morego: Senior VP of Investor Relations who raises pointed questions about the validity of Dane's predictions
    Silence: Brief mention as an Uber driver
    C-Level Executives: Present as an audience during the presentation

    World Description:
    The story takes place in a contemporary corporate setting, likely a financial technology company. The world appears to be our modern one, with familiar elements like:
    Major tech companies (Tesla, Google, Apple, Microsoft)
    Stock market and financial systems
    Modern technology (neural networks, predictive analytics)
    Urban environment with rideshare services like Uber
    Corporate hierarchy and office culture

    The story creates tension between the familiar corporate world and the potential for an unprecedented financial catastrophe, blending elements of technical thriller with workplace drama. The setting feels grounded in reality but hints at potentially apocalyptic economic consequences.
    """


def run_book_generation(
    config: Optional[AppConfig] = None,
    custom_prompt: Optional[str] = None,
    use_autogen2: Optional[bool] = None,
) -> None:
    """Run the complete book generation process

    Args:
        config: Application configuration (uses defaults if not provided)
        custom_prompt: Optional custom story prompt to use instead of default
        use_autogen2: Whether to use AutoGen 2.0 API (default: auto-detect)
    """
    logger = get_logger("main")
    logger.info("=" * 50)
    logger.info("Starting Book Generator")
    logger.info("=" * 50)

    # Get or create configuration
    if config is None:
        config = get_app_config()

    # Auto-detect AutoGen 2.0 if not specified
    if use_autogen2 is None:
        use_autogen2 = USE_AUTOGEN2_DEFAULT

    logger.info(f"Using AutoGen 2.0: {use_autogen2}")
    logger.info(f"Using LLM provider: {config.provider}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Number of chapters: {config.default_num_chapters}")

    # Get the story prompt
    initial_prompt = custom_prompt or get_initial_prompt()
    num_chapters = config.default_num_chapters

    try:
        # Phase 1: Generate Outline
        logger.info("\n" + "=" * 50)
        logger.info("Phase 1: Generating Book Outline")
        logger.info("=" * 50)

        # AutoGen 2.0 path: use model_client
        outline_agent_config = config.get_agent_config_for_role("outline_creator")
        model_client = config.create_model_client_for_role("outline_creator")

        outline_agents = BookAgents(
            {"model_client": model_client},
            num_chapters=num_chapters,
            initial_prompt=initial_prompt,
            use_autogen2=True,
        )


        agents = outline_agents.create_agents(initial_prompt, num_chapters)

        outline_gen = OutlineGenerator(
            agents,
            outline_agent_config.to_dict() if not use_autogen2 else {"model_client": model_client},
            use_autogen2=use_autogen2,
        )
        outline = outline_gen.generate_outline(initial_prompt, num_chapters)

        if not outline:
            logger.error("Failed to generate outline")
            sys.exit(1)

        # Save outline to file for inspection
        outline_path = os.path.join(config.output_dir, "outline.txt")
        save_outline_to_file(outline, outline_path, initial_prompt)
        logger.info(f"✓ Outline saved to: {outline_path}")

        # Log outline at INFO level for visibility
        logger.info(f"\nGenerated Outline ({len(outline)} chapters):")
        for chapter in outline:
            logger.info(f"\nChapter {chapter['chapter_number']}: {chapter['title']}")
            logger.info("-" * 50)
            logger.info(f"Prompt:\n{chapter['prompt']}")  # Changed from debug to info

        # Phase 2: Generate Book
        logger.info("\n" + "=" * 50)
        logger.info("Phase 2: Generating Book Chapters")
        logger.info("=" * 50)

        # Use creative model for writer, planning model for other agents
        writer_config = config.get_agent_config_for_role("writer")

        model_client = config.create_model_client_for_role("writer")
        book_agents = BookAgents(
            {"model_client": model_client},
            outline=outline,
            num_chapters=num_chapters,
            use_autogen2=True,
        )


        agents_with_context = book_agents.create_agents(initial_prompt, num_chapters)

        book_gen = BookGenerator(
            agents_with_context,
            writer_config.to_dict() if not use_autogen2 else {"model_client": model_client},
            outline,
            output_dir=config.output_dir,
            use_autogen2=use_autogen2,
            emergency_generation_enabled=config.emergency_generation_enabled,
        )

        book_gen.generate_book(outline)

        # Phase 3: Summary
        logger.info("\n" + "=" * 50)
        logger.info("Book Generation Complete!")
        logger.info("=" * 50)

        stats = book_gen.get_book_stats()
        logger.info("\nStatistics:")
        logger.info(f"Total chapters generated: {stats['total_chapters']}")
        logger.info(f"Total word count: {stats['total_words']:,}")

        if stats['chapters']:
            avg_words = stats['total_words'] // stats['total_chapters']
            logger.info(f"Average words per chapter: {avg_words:,}")

        logger.info(f"\nOutput saved to: {config.output_dir}")

    except Exception as e:
        logger.exception("Book generation failed")
        logger.error(f"Error: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Book Generator - Generate complete books using collaborative AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py

  # Load story from markdown file
  python main.py --prompt story.md

  # Specify number of chapters
  python main.py --prompt story.md --chapters 10

  # Use specific provider
  python main.py --provider openai --prompt story.md

  # Force AutoGen 2.0 or legacy mode
  python main.py --autogen2
  python main.py --legacy-autogen
        """
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Path to a markdown/text file containing the story outline"
    )

    parser.add_argument(
        "--chapters", "-c",
        type=int,
        default=None,
        help="Number of chapters to generate (overrides .env setting)"
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "openai", "azure"],
        help="LLM provider to use (overrides .env setting)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory (overrides .env setting)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--autogen2",
        action="store_true",
        help="Use AutoGen 2.0 API (if available)"
    )

    parser.add_argument(
        "--legacy-autogen",
        action="store_true",
        help="Use legacy AutoGen 0.2 API"
    )

    return parser.parse_args()


def main():
    """Main entry point with CLI support"""
    # Parse arguments first
    args = parse_arguments()

    # Setup logging early
    setup_logging(level=args.log_level)
    logger = get_logger("main")

    # Determine AutoGen version to use
    use_autogen2 = None
    if args.autogen2 and args.legacy_autogen:
        logger.error("Cannot specify both --autogen2 and --legacy-autogen")
        sys.exit(1)
    elif args.autogen2:
        use_autogen2 = True
    elif args.legacy_autogen:
        use_autogen2 = False

    # Load prompt from file if specified
    custom_prompt = None
    if args.prompt:
        try:
            custom_prompt = load_prompt_from_file(args.prompt)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to load prompt file: {e}")
            sys.exit(1)

    # Build configuration
    config_kwargs = {}
    if args.provider:
        config_kwargs["provider"] = args.provider
    if args.output:
        config_kwargs["output_dir"] = args.output

    try:
        config = get_app_config(**config_kwargs)

        # Override chapters if specified
        if args.chapters:
            config.default_num_chapters = args.chapters
            logger.info(f"Overriding chapter count: {args.chapters}")

        # Run book generation
        run_book_generation(
            config=config,
            custom_prompt=custom_prompt,
            use_autogen2=use_autogen2,
        )
    except KeyboardInterrupt:
        logger.info("\nGeneration interrupted by user")
        sys.exit(0)
    except Exception:
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
