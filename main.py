"""Main script for running the book generation system"""
from dotenv import load_dotenv
from config import get_config
from agents import BookAgents
from book_generator import BookGenerator
from outline_generator import OutlineGenerator

load_dotenv()

def main():
    # Get configuration
    agent_config = get_config()

    
    # Initial prompt for the book
    initial_prompt = open("story_prose.txt", "r").read()
    
    # Number of chapters to generate
    num_chapters = 10
    
    print("Generate Book")
    
    # Create agents
    outline_agents = BookAgents(agent_config)
    agents = outline_agents.create_agents(initial_prompt, num_chapters)
    
    # Generate the outline
    outline_gen = OutlineGenerator(agents, agent_config)
    print("Generating book outline...")
    outline = outline_gen.generate_outline(initial_prompt, num_chapters)
    
    # Create new agents with outline context
    book_agents = BookAgents(agent_config, outline=outline)
    agents_with_context = book_agents.create_agents(initial_prompt, num_chapters)
    
    # Initialize book generator with contextual agents
    book_gen = BookGenerator(agents_with_context, agent_config, outline)
        
    # Print the generated outline
    print("\nGenerated Outline:")
    for chapter in outline:
        print(f"\nChapter {chapter['chapter_number']}: {chapter['title']}")
        print("-" * 50)
        print(chapter['prompt'])
        
        # Save the outline for reference
        print("\nSaving outline to file...")
        with open("book_output/outline.txt", "w") as f:
            for chapter in outline:
                f.write(f"\nChapter {chapter['chapter_number']}: {chapter['title']}\n")
                f.write("-" * 50 + "\n")
                f.write(chapter['prompt'] + "\n")
                f.flush()
        
        # Generate the book using the outline
        print("\nGenerating book chapters...")
        if outline:
            book_gen.generate_book(outline)
        else:
            print("Error: No outline was generated.")

if __name__ == "__main__":
    main()