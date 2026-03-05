"""Pydantic models for book generation data validation"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class Chapter(BaseModel):
    """Represents a single chapter with validation"""
    chapter_number: int = Field(..., ge=1, description="Chapter number starting from 1")
    title: str = Field(..., min_length=1, max_length=200, description="Chapter title")
    prompt: str = Field(..., min_length=10, description="Chapter generation prompt")
    
    @field_validator('chapter_number')
    @classmethod
    def validate_chapter_number(cls, v: int) -> int:
        if v < 1:
            raise ValueError('Chapter number must be at least 1')
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Title cannot be empty')
        return v
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10:
            raise ValueError('Prompt must be at least 10 characters')
        return v


class Outline(BaseModel):
    """Represents a complete book outline with validation"""
    chapters: List[Chapter] = Field(..., min_length=1, description="List of chapters")
    total_chapters: int = Field(..., ge=1, description="Expected total number of chapters")
    
    @model_validator(mode='after')
    def validate_outline(self):
        """Validate chapter numbering is sequential and complete"""
        chapters = self.chapters
        total = self.total_chapters
        
        if len(chapters) != total:
            raise ValueError(f'Expected {total} chapters, got {len(chapters)}')
        
        # Check for sequential numbering
        expected_numbers = set(range(1, total + 1))
        actual_numbers = {ch.chapter_number for ch in chapters}
        
        missing = expected_numbers - actual_numbers
        duplicates = len(chapters) - len(actual_numbers)
        
        if missing:
            raise ValueError(f'Missing chapter numbers: {sorted(missing)}')
        
        if duplicates:
            raise ValueError(f'Found {duplicates} duplicate chapter number(s)')
        
        # Sort chapters by number
        self.chapters = sorted(chapters, key=lambda x: x.chapter_number)
        
        return self
    
    def get_chapter(self, chapter_number: int) -> Optional[Chapter]:
        """Get a chapter by number"""
        for chapter in self.chapters:
            if chapter.chapter_number == chapter_number:
                return chapter
        return None
    
    def get_next_chapter(self, chapter_number: int) -> Optional[Chapter]:
        """Get the next chapter"""
        return self.get_chapter(chapter_number + 1)
    
    def is_last_chapter(self, chapter_number: int) -> bool:
        """Check if this is the last chapter"""
        return chapter_number >= self.total_chapters


class ChapterContent(BaseModel):
    """Represents generated chapter content with validation"""
    chapter_number: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=100, description="Generated chapter content")
    word_count: int = Field(default=0, ge=0)
    
    @model_validator(mode='after')
    def calculate_word_count(self):
        """Calculate word count from content"""
        if self.content:
            self.word_count = len(self.content.split())
        return self
    
    def validate_length(self, min_words: int = 3000) -> bool:
        """Validate that chapter meets minimum word count"""
        return self.word_count >= min_words


class WorldElement(BaseModel):
    """Represents a world-building element"""
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=10)
    location_type: str = Field(default="location", description="Type of element (location, object, etc.)")
    first_appearance: Optional[int] = Field(default=None, ge=1, description="First chapter where this appears")
    recurring: bool = Field(default=False, description="Whether this element appears multiple times")


class CharacterDevelopment(BaseModel):
    """Represents character development tracking"""
    character_name: str = Field(..., min_length=1)
    development_notes: List[str] = Field(default_factory=list)
    first_appearance: Optional[int] = Field(default=None, ge=1)
    arc_description: Optional[str] = Field(default=None)
    
    def add_development(self, note: str) -> None:
        """Add a development note for this character"""
        self.development_notes.append(note)


class GenerationResult(BaseModel):
    """Represents the result of a generation attempt"""
    success: bool
    chapter_number: int
    content: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    generation_time_seconds: float = 0.0


class StoryArc(BaseModel):
    """Represents a high-level story arc"""
    major_plot_points: List[str] = Field(default_factory=list)
    character_arcs: dict = Field(default_factory=dict)
    story_beats: List[str] = Field(default_factory=list)
    key_transitions: List[str] = Field(default_factory=list)
