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
    initial_prompt = """
    Create a story in my established writing style with these key elements:
    Its important that it has several key storylines that intersect and influence each other. The story should be set in a world within the Minecraft game. The protagonist is a small boy named Matthew who has just fallen asleep in front of his computer while playing his favourite computer game Minecraft . When Matthew wakes up he is in the game, stuck he need to solves puzzles and travel through the Minecraft world to find a portal to get home.

    The piece is written in third-person limited perspective, following Matthew thoughts and experiences. The prose is direct and technical when describing the protagonist's work, but becomes more introspective during personal moments. The author employs a mix of dialogue and internal monologue, with particular attention to time progression and how Matthew might get home.
    Story Arch:

    Setup: Matthew falls a sleep late at night while playing Minecraft
    Initial Conflict: Matthew awakens in the game and realizes he is stuck
    Rising Action: Matthew must solve puzzles and navigate the Minecraft world
    Climax: Matthew finds the portal to return home
    Tension Point: Matthew must decide whether to stay in the game or return to reality

    Characters:

    Matthew: The protagonist; a small boy who love to play computer games online with his friends.
    Maks: The best freind of Matthew is also a gamer and helps him navigate the Minecraft world.
    Pillager: Pillagers are hostile mobs armed with crossbows found in wandering patrols, in pillager outposts, or as participants in raids. They attack by firing arrows at the player.
    Mob: A mob is an AI-driven game entity resembling a living creature. Beside its common meaning, the term "mob" is short for "mobile entity".[1] All mobs can be attacked and hurt (from falling, attacked by a player or another mob, falling into the void, hit by an arrow, etc), and have some form of voluntary movement. Different types of mobs often have unique AI and drop good or bad loot depending on the mob that was killed.
    Animal: The term Animal refers to a category of mobs that are mainly based on real life animals. Mobs mentioned on this page are classified as Animal or WaterAnimal in the game code. Many other mobs and even some blocks are also based on real life animals, but are not treated as such in Minecraft. Most of these mobs are also called animals in many advancements. The blocks are not listed on this page. Animals are usually either passive (fleeing) or neutral (fighting), with the only exception being the hostile hoglin.

    World Description:
    The story takes place in the Minecraft game. The world is a blocky, procedurally-generated 3D world where players can explore, build, and interact with the environment. The world is populated by various creatures, including animals, mobs, and villagers. The landscape features a variety of biomes, such as forests, deserts, mountains, and oceans. Players can mine resources, craft items, and build structures using blocks. The world is infinite in size, with no set goals or objectives, allowing players to create their own adventures.

    The story creates tension the human characters and the creatures in the Minecraft world. The protagonist must navigate the world and interact with its inhabitants to find a way home. The story explores themes of friendship, survival, and the blurring of reality and fantasy. The protagonist's journey is both physical and emotional, as he confronts challenges and makes difficult decisions. The story is a mix of adventure, mystery, and drama, with elements of humor and suspense. The world of Minecraft is a rich and vibrant setting that adds depth and complexity to the narrative.
    """
    num_chapters = 15
    
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