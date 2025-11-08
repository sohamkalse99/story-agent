import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class StoryRequest:
    theme: str
    age_group: str = "5-10"
    length: str = "medium"  # short, medium, long
    moral: Optional[str] = None

class StoryGenerator:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.max_retries = 3
        self.categories = {
            "adventure": self._generate_adventure_story,
            "fairy_tale": self._generate_fairy_tale,
            "educational": self._generate_educational_story,
            "animal": self._generate_animal_story,
            "fantasy": self._generate_fantasy_story
        }

    def generate_story(self, request: StoryRequest) -> str:
        """Generate a story based on the request with category-specific generation."""
        # First, determine the category of the story
        category = self._determine_category(request)
        print(f"🎭 Selected story category: {category.replace('_', ' ').title()}")
        
        # Then generate using the appropriate strategy
        return self.categories[category](request)
    
    def _determine_category(self, request: StoryRequest) -> str:
        """Determine the most appropriate category for the story."""
        prompt = f"""
        Based on the following story request, select the most appropriate category:
        - Theme: {request.theme}
        - Age Group: {request.age_group}
        - Moral: {request.moral if request.moral else 'Not specified'}
        
        Available categories:
        1. adventure - Exciting journeys and challenges
        2. fairy_tale - Magical stories with moral lessons
        3. educational - Stories that teach specific concepts
        4. animal - Stories centered around animal characters
        5. fantasy - Imaginative stories with magical elements
        
        Respond with ONLY the category name from the list above.
        """
        
        response = self._call_llm(prompt, temperature=0.2, max_tokens=50)
        category = response.strip().lower()
        
        # Default to adventure if the response isn't a valid category
        return category if category in self.categories else "adventure"
    
    def _build_common_prompt(self, request: StoryRequest, category: str) -> str:
        """Build the common parts of the prompt for all story types."""
        return f"""
        You are a master storyteller creating a {category.replace('_', ' ')} story 
        for children aged {request.age_group}.
        
        STORY DETAILS:
        - Theme: {request.theme}
        - Length: {request.length}
        {f'- Moral: {request.moral}' if request.moral else ''}
        
        """
    
    def _generate_adventure_story(self, request: StoryRequest) -> str:
        """Generate an adventure story with exciting challenges."""
        prompt = self._build_common_prompt(request, "adventure") + """
        Create an exciting adventure story with these elements:
        1. A brave main character facing challenges
        2. A clear quest or mission
        3. Exciting obstacles to overcome
        4. A satisfying resolution
        5. Dialogue that moves the story forward
        
        Make it engaging and full of action, perfect for young adventurers!
        """
        return self._call_llm(prompt)
    
    def _generate_fairy_tale(self, request: StoryRequest) -> str:
        """Generate a magical fairy tale with a moral lesson."""
        moral = request.moral or "the importance of kindness"
        prompt = self._build_common_prompt(request, "fairy tale") + f"""
        Create a magical fairy tale that teaches "{moral}" with these elements:
        1. A magical setting or elements
        2. Memorable characters (good and bad)
        3. A clear problem and solution
        4. A moral lesson about {moral}
        5. A happy ending with justice served
        
        Use rich descriptions and make it feel like a classic fairy tale.
        """
        return self._call_llm(prompt)
    
    def _generate_educational_story(self, request: StoryRequest) -> str:
        """Generate a story that teaches something educational."""
        topic = request.theme or "an important life lesson"
        prompt = self._build_common_prompt(request, "educational") + f"""
        Create an educational story that teaches about "{topic}" with these elements:
        1. Clear educational content about {topic}
        2. Relatable characters learning something new
        3. Real-world applications of the lesson
        4. Questions or thinking points for the reader
        
        Make it informative yet entertaining for a {request.age_group}-year-old.
        """
        return self._call_llm(prompt)
    
    def _generate_animal_story(self, request: StoryRequest) -> str:
        """Generate a story with animal characters."""
        prompt = self._build_common_prompt(request, "animal") + """
        Create a heartwarming story with animal characters that includes:
        1. Animal characters with distinct personalities
        2. A setting in nature
        3. A problem that needs solving
        4. Teamwork or clever thinking
        5. A positive message about friendship or cooperation
        
        Make the animals talk and have human-like emotions.
        """
        return self._call_llm(prompt)
    
    def _generate_fantasy_story(self, request: StoryRequest) -> str:
        """Generate a fantasy story with magical elements."""
        prompt = self._build_common_prompt(request, "fantasy") + """
        Create a magical fantasy story with these elements:
        1. A unique magical world or ability
        2. A hero's journey
        3. Magical creatures or elements
        4. A clear conflict and resolution
        5. A moral or lesson learned
        
        Let your imagination run wild with magical possibilities!
        """
        return self._call_llm(prompt)
    
    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Call the language model with error handling and retries."""
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=min(temperature, 1.0),  # Ensure temperature doesn't exceed 1.0
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed after {self.max_retries} attempts: {str(e)}")
                continue

class StoryJudge:
    """Evaluates and provides feedback on generated stories."""
    
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        
    def evaluate_story(self, story: str, request: StoryRequest) -> Dict:
        """Evaluate the story and provide feedback."""
        # First, ensure we have a story to evaluate
        if not story.strip():
            return self._create_default_evaluation("No story was generated to evaluate.")
            
        try:
            prompt = self._build_evaluation_prompt(story, request)
            evaluation = self._call_llm(prompt)
            return self._parse_evaluation(evaluation)
        except Exception as e:
            return self._create_default_evaluation(f"Error during evaluation: {str(e)}")
    
    def _build_evaluation_prompt(self, story: str, request: StoryRequest) -> str:
        """Build the prompt for story evaluation."""
        return f"""
        You are a children's story judge. Your task is to evaluate the following story.
        
        STORY REQUEST:
        - Theme: {request.theme}
        - Age Group: {request.age_group}
        - Desired Length: {request.length}
        {f'- Moral: {request.moral}' if request.moral else ''}
        
        STORY TO EVALUATE:
        ""{story}""
        
        YOUR TASK:
        1. Read the story carefully
        2. Rate it on the following criteria (1-5, where 5 is best)
        3. Provide specific feedback and suggestions
        
        CRITERIA:
        1. Age-appropriateness for {request.age_group}
        2. Engagement and creativity
        3. Story structure (beginning, middle, end)
        4. Educational value
        5. Positive messaging
        6. Language and clarity
        
        RESPONSE FORMAT (valid JSON only, no other text):
        {{
            "scores": {{
                "age_appropriate": 4,
                "engagement": 3,
                "structure": 4,
                "educational_value": 2,
                "positive_messaging": 5,
                "language_clarity": 4
            }},
            "overall_score": 3.7,
            "feedback": "The story is engaging but could be more educational.",
            "suggestions": [
                "Add more educational elements about the theme",
                "Consider adding more dialogue between characters"
            ]
        }}"""
    
    def _call_llm(self, prompt: str) -> str:
        """Call the language model with error handling."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates children's stories. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error calling language model: {str(e)}")
    
    def _create_default_evaluation(self, message: str) -> Dict:
        """Create a default evaluation with an error message."""
        return {
            "scores": {},
            "overall_score": 0,
            "feedback": message,
            "suggestions": ["Please try generating the story again."]
        }
    
    def _parse_evaluation(self, evaluation_text: str) -> Dict:
        """Parse the evaluation from text to a dictionary."""
        try:
            # First, try to find JSON in the response
            json_start = evaluation_text.find('{')
            json_end = evaluation_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = evaluation_text[json_start:json_end]
                evaluation = json.loads(json_str)
                
                # Ensure all required fields exist
                if not all(key in evaluation for key in ["scores", "overall_score", "feedback", "suggestions"]):
                    return self._create_default_evaluation("Incomplete evaluation received.")
                
                return evaluation
            else:
                return self._create_default_evaluation("Could not find valid evaluation in the response.")
                
        except json.JSONDecodeError as e:
            return self._create_default_evaluation(f"Failed to parse evaluation: {str(e)}. Response: {evaluation_text[:200]}")
        except Exception as e:
            return self._create_default_evaluation(f"Error processing evaluation: {str(e)}")

def get_user_input() -> StoryRequest:
    """Get story requirements from the user."""
    print("\n🎭 Welcome to the Bedtime Story Generator! 🎭\n")
    
    theme = input("What's the theme or main idea for your story? (e.g., a brave little mouse) ").strip()
    age_group = input("What age group is this for? (5-10) ").strip() or "5-10"
    length = input("How long should the story be? (short/medium/long) ").strip().lower()
    length = length if length in ["short", "medium", "long"] else "medium"
    moral = input("Is there a specific moral or lesson? (Press Enter to skip) ").strip()
    
    return StoryRequest(
        theme=theme,
        age_group=age_group,
        length=length,
        moral=moral if moral else None
    )

def get_user_feedback(story: str) -> Dict[str, str]:
    """Get feedback from the user about the generated story."""
    print("\n🌟 How did you like the story? 🌟")
    print("1. I love it! 🎉")
    print("2. It's okay, but could be better")
    print("3. I'd like to request changes")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            return {"status": "loved", "feedback": ""}
        elif choice == '2':
            feedback = input("\nWhat would you like to improve? ")
            return {"status": "needs_improvement", "feedback": feedback}
        elif choice == '3':
            feedback = input("\nWhat changes would you like to make? ")
            return {"status": "needs_changes", "feedback": feedback}
        else:
            print("Please enter a number between 1 and 3")

def improve_story(story: str, feedback: str, generator: StoryGenerator, request: StoryRequest) -> str:
    """Improve the story based on user feedback."""
    print("\n✨ Working on improvements...")
    
    prompt = f"""
    Please improve the following story based on this feedback: "{feedback}"
    
    Original Story:
    {story}
    
    Improved Story:"""
    
    return generator._call_llm(prompt)

def main():
    # Initialize components
    generator = StoryGenerator()
    judge = StoryJudge()
    
    try:
        # Get user input
        request = get_user_input()
        
        # Generate initial story
        print("\n✨ Crafting your magical story...")
        story = generator.generate_story(request)
        
        # Main interaction loop
        while True:
            # Evaluate story
            print("🔍 Evaluating story quality...")
            evaluation = judge.evaluate_story(story, request)
            
            # Display results
            print("\n📖 Your Bedtime Story 📖\n")
            print(story)
            
            # Show quality feedback if needed
            if evaluation["overall_score"] < 3:
                print("\n⚠️  Note: The story might need improvement. Here's some feedback:")
                print(f"- {evaluation['feedback']}")
                if evaluation['suggestions']:
                    print("\nSuggestions for improvement:")
                    for i, suggestion in enumerate(evaluation['suggestions'][:3], 1):
                        print(f"{i}. {suggestion}")
            
            # Get user feedback
            feedback = get_user_feedback(story)
            
            if feedback["status"] == "loved":
                print("\n🎉 We're so glad you loved the story! Sweet dreams! 🌟\n")
                break
                
            elif feedback["status"] in ["needs_improvement", "needs_changes"]:
                if not feedback["feedback"].strip():
                    print("\nPlease provide some feedback for improvement.")
                    continue
                    
                story = improve_story(story, feedback["feedback"], generator, request)
                print("\n✨ Here's the improved version!")
                
            else:
                print("\n🌟 The End! Sweet dreams! 🌟\n")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for using the Bedtime Story Generator!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please try again or check your API key and internet connection.")

if __name__ == "__main__":
    main()