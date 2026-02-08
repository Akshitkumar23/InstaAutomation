#!/usr/bin/env python3
"""
Instagram AI Agent - Autonomous Content Publishing System

This module implements a fully automated, end-to-end Instagram content publishing system
that plans, generates, executes, evaluates, and improves Instagram posts without human intervention.
"""

import os
import json
import logging
import datetime
import hashlib
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from ai_instagram_agent.model_runtime import ModelRuntime

BASE_DIR = Path(__file__).resolve().parent


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    alt = BASE_DIR / path
    return alt if alt.exists() else path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instagram_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PostContent:
    """Data structure for Instagram post content"""
    day: str
    selected_topic: str
    image_prompt: str
    caption: str
    hashtags: List[str]
    post_time: str
    training_update: Optional[Dict] = None


class InstagramAIAgent:
    """Autonomous Instagram AI Agent for content publishing"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the Instagram AI Agent
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = _resolve_config_path(config_path)
        self.config = self._load_config()
        self.config.setdefault("__base_dir", str(self.config_path.parent))
        self.memory_path = Path("agent_memory.json")
        self.memory = self._load_memory()
        self.model_runtime = ModelRuntime(self.config)
        
        # Content type rotation
        self.content_types = [
            "educational", "motivational", "system-based", 
            "habit-based", "insight-based"
        ]
        
        # Performance tracking
        self.performance_history = []
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            with self.config_path.open('r', encoding='utf-8') as f:
                config = json.load(f)
            config["__base_dir"] = str(self.config_path.parent)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "api_credentials": {
                "instagram_access_token": "",
                "instagram_user_id": "",
                "filestack_api_key": ""
            },
            "posting_schedule": {
                "daily_time": "14:00",  # 2:00 PM
                "timezone": "Asia/Calcutta"
            },
            "content_preferences": {
                "max_caption_length": 150,
                "min_hashtags": 15,
                "max_hashtags": 25,
                "preferred_emojis": ["✨", "🚀", "💡", "🎯", "🔥"],
                "banned_hashtags": []
            },
            "performance_thresholds": {
                "engagement_decline_threshold": 3,
                "min_engagement_score": 50
            }
        }
    
    def _load_memory(self) -> Dict:
        """Load agent memory from file"""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
        
        return {
            "topics_used": [],
            "content_history": [],
            "performance_metrics": [],
            "topic_preferences": {},
            "hashtag_performance": {},
            "caption_patterns": {}
        }
    
    def _save_memory(self):
        """Save agent memory to file"""
        try:
            with open(self.memory_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def _get_current_day(self) -> str:
        """Get current day in YYYY-MM-DD format"""
        return datetime.datetime.now().strftime("%Y-%m-%d")
    
    def _is_topic_used_recently(self, topic: str, days: int = 30) -> bool:
        """Check if topic was used in the last N days"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        recent_topics = [
            entry["selected_topic"] 
            for entry in self.memory["content_history"]
            if datetime.datetime.fromisoformat(entry["day"]) > cutoff_date
        ]
        return topic.lower() in [t.lower() for t in recent_topics]
    
    def _select_content_topic(self) -> str:
        """
        Step 1: CONTENT STRATEGY DECISION
        Decide exactly ONE post topic for today
        """
        logger.info("Step 1: Selecting content topic")
        
        # Define potential topics
        potential_topics = [
            "Productivity hacks for remote workers",
            "Morning routine optimization",
            "Digital minimalism principles",
            "Goal setting strategies",
            "Time management techniques",
            "Mindfulness in daily life",
            "Creative problem solving",
            "Learning new skills efficiently",
            "Building positive habits",
            "Overcoming procrastination",
            "Work-life balance tips",
            "Digital detox benefits",
            "Focus and concentration methods",
            "Decision making frameworks",
            "Personal growth mindset"
        ]
        
        # Filter out recently used topics
        available_topics = [
            topic for topic in potential_topics 
            if not self._is_topic_used_recently(topic)
        ]
        
        if not available_topics:
            # If all topics used recently, use least recent
            available_topics = potential_topics
        
        # Select random topic from available
        selected_topic = random.choice(available_topics)
        
        # Store in memory
        self.memory["topics_used"].append({
            "topic": selected_topic,
            "date": self._get_current_day()
        })
        
        logger.info(f"Selected topic: {selected_topic}")
        return selected_topic
    
    def _generate_image_prompt(self, topic: str) -> str:
        """
        Step 2: IMAGE PROMPT GENERATION
        Generate detailed IMAGE_PROMPT for 1080x1080 Instagram feed post
        """
        logger.info("Step 2: Generating image prompt")
        
        # Base prompt structure
        base_prompt = (
            f"Create a 1080x1080 Instagram feed post image about '{topic}'. "
            "Clean, modern, Instagram-native aesthetic. Minimal clutter. "
            "High contrast and readable typography. Emotionally engaging. "
            "No brand logos or copyrighted characters. "
            "Professional and inspiring visual style."
        )
        
        # Add visual elements based on topic
        visual_elements = {
            "Productivity": "Include abstract productivity symbols like checkmarks, calendars, or progress bars",
            "Morning": "Use warm morning colors and sunrise imagery",
            "Digital": "Incorporate clean digital/tech elements",
            "Goal": "Include target or milestone visual metaphors",
            "Time": "Use clock or hourglass elements",
            "Mindfulness": "Include calming nature or meditation elements",
            "Creative": "Use artistic and colorful abstract elements",
            "Learning": "Include book or brain imagery",
            "Habits": "Show habit tracking or routine elements",
            "Procrastination": "Use contrast between action and inaction",
            "Balance": "Show work-life balance visual metaphors",
            "Detox": "Include fresh, clean, and refreshing elements",
            "Focus": "Use sharp, clear visual elements",
            "Decision": "Include choice or path imagery",
            "Growth": "Show growth or development visual metaphors"
        }
        
        # Find relevant visual elements
        for key, elements in visual_elements.items():
            if key.lower() in topic.lower():
                base_prompt += f" {elements}"
                break
        
        logger.info(f"Generated image prompt: {base_prompt[:100]}...")
        return base_prompt
    
    def _generate_caption_and_hashtags(self, topic: str) -> Tuple[str, List[str]]:
        """
        Step 3: CAPTION AND HASHTAG GENERATION
        Generate high-engagement caption and hashtags
        """
        logger.info("Step 3: Generating caption and hashtags")

        try:
            text_result = self.model_runtime.generate_caption_and_hashtags(topic)
            return text_result.caption, text_result.hashtags
        except Exception as exc:
            logger.warning(f"Text model failed, falling back to templates: {exc}")
        
        # Generate caption
        caption_templates = [
            f"🚀 {topic}! Here's how you can implement this in your daily life:\n\n"
            "✨ Key insights:\n"
            "• Point 1\n"
            "• Point 2\n"
            "• Point 3\n\n"
            "💡 Try this today and share your experience!\n\n"
            "👇 What's your biggest challenge with this topic?",
            
            f"💡 Ever wondered about {topic}? Let's break it down:\n\n"
            "🎯 Why it matters:\n"
            "- Reason 1\n"
            "- Reason 2\n"
            "- Reason 3\n\n"
            "🔥 Pro tip: Start small and build consistency!\n\n"
            "💬 Agree? Drop a comment below!",
            
            f"✨ {topic} can transform your life if you approach it right:\n\n"
            "🚀 Implementation steps:\n"
            "1. Step one\n"
            "2. Step two\n"
            "3. Step three\n\n"
            "🌟 Remember: Progress > perfection!\n\n"
            "🔄 Share this with someone who needs it!"
        ]
        
        caption = random.choice(caption_templates)
        
        # Ensure caption length constraint
        if len(caption.split()) > self.config["content_preferences"]["max_caption_length"]:
            caption = caption[:self.config["content_preferences"]["max_caption_length"]]
        
        # Generate hashtags
        base_hashtags = [
            "productivity", "motivation", "lifestyle", "success", "growth",
            "mindset", "habits", "goals", "inspiration", "selfimprovement",
            "dailyroutine", "personalgrowth", "lifetips", "mindfulness", "focus",
            "creativity", "learning", "development", "achievement", "positivity"
        ]
        
        topic_hashtags = [
            hashtag for hashtag in base_hashtags 
            if hashtag in topic.lower() or len(topic.split()) > 3
        ]
        
        # Add topic-specific hashtags
        topic_words = topic.lower().split()
        for word in topic_words:
            if len(word) > 3 and word not in ["about", "with", "from", "into"]:
                topic_hashtags.append(word.replace(" ", ""))
        
        # Remove banned hashtags and duplicates
        banned = self.config["content_preferences"]["banned_hashtags"]
        available_hashtags = [
            f"#{tag}" for tag in topic_hashtags 
            if tag not in banned and len(tag) > 2
        ]
        
        # Ensure hashtag count constraints
        min_h = self.config["content_preferences"]["min_hashtags"]
        max_h = self.config["content_preferences"]["max_hashtags"]
        
        if len(available_hashtags) < min_h:
            # Add general hashtags if needed
            general_hashtags = ["dailyinspiration", "lifecoach", "successmindset"]
            available_hashtags.extend([f"#{tag}" for tag in general_hashtags])
        
        selected_hashtags = random.sample(
            available_hashtags, 
            min(max_h, len(available_hashtags))
        )
        
        logger.info(f"Generated caption and {len(selected_hashtags)} hashtags")
        return caption, selected_hashtags
    
    def _prepare_execution_output(self, content: PostContent) -> Dict:
        """
        Step 4: EXECUTION COMMAND OUTPUT
        Prepare execution-ready outputs for downstream modules
        """
        logger.info("Step 4: Preparing execution output")
        
        execution_data = {
            "image_generation": {
                "prompt": content.image_prompt,
                "dimensions": "1080x1080",
                "style": "clean_modern_instagram"
            },
            "instagram_publishing": {
                "caption": content.caption,
                "hashtags": content.hashtags,
                "post_time": content.post_time,
                "media_type": "image"
            },
            "metadata": {
                "day": content.day,
                "topic": content.selected_topic,
                "content_type": self._classify_content_type(content.selected_topic)
            }
        }
        
        return execution_data
    
    def _get_post_time(self) -> str:
        """Get optimized posting time"""
        return self.config["posting_schedule"]["daily_time"]
    
    def _classify_content_type(self, topic: str) -> str:
        """Classify content type based on topic"""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["learn", "skill", "knowledge", "study"]):
            return "educational"
        elif any(word in topic_lower for word in ["motiv", "inspir", "encourag"]):
            return "motivational"
        elif any(word in topic_lower for word in ["system", "process", "method"]):
            return "system-based"
        elif any(word in topic_lower for word in ["habit", "routine", "daily"]):
            return "habit-based"
        else:
            return "insight-based"
    
    def _update_training_memory(self, content: PostContent, engagement_score: Optional[int] = None):
        """
        Step 7: SELF-TRAINING AND ADAPTATION
        Update internal performance memory and adapt
        """
        logger.info("Step 7: Updating training memory")
        
        # Store content in history
        self.memory["content_history"].append(asdict(content))
        
        # Update topic preferences
        topic_type = self._classify_content_type(content.selected_topic)
        if topic_type not in self.memory["topic_preferences"]:
            self.memory["topic_preferences"][topic_type] = 0
        self.memory["topic_preferences"][topic_type] += 1
        
        # Update hashtag performance if engagement score provided
        if engagement_score:
            for hashtag in content.hashtags:
                if hashtag not in self.memory["hashtag_performance"]:
                    self.memory["hashtag_performance"][hashtag] = []
                self.memory["hashtag_performance"][hashtag].append(engagement_score)
        
        # Check for engagement decline
        if len(self.performance_history) >= 7:
            recent_scores = self.performance_history[-7:]
            if all(score < self.config["performance_thresholds"]["min_engagement_score"] 
                   for score in recent_scores[-3:]):
                logger.warning("Engagement decline detected - entering exploration mode")
                # Implementation would include experimenting with new patterns
        
        self._save_memory()
        logger.info("Training memory updated")
    
    def execute_daily_cycle(self, engagement_score: Optional[int] = None) -> PostContent:
        """
        Execute one complete daily cycle of the Instagram AI Agent
        
        Args:
            engagement_score: Optional engagement score from previous post
            
        Returns:
            PostContent: Generated content for today
        """
        logger.info("=" * 50)
        logger.info("Starting daily Instagram AI Agent cycle")
        logger.info("=" * 50)
        
        day = self._get_current_day()
        
        # Step 1: Content Strategy Decision
        selected_topic = self._select_content_topic()
        
        # Step 2: Image Prompt Generation
        image_prompt = self._generate_image_prompt(selected_topic)
        
        # Step 2.5: Image Generation (FIXED - Now calling image generation!)
        logger.info("Step 2.5: Generating image")
        image_path = None
        if self.model_runtime.should_generate_images():
            try:
                image_path = self.model_runtime.generate_image(image_prompt)
                logger.info(f"Image generated: {image_path}")
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                image_path = None
        
        # Step 3: Caption and Hashtag Generation
        caption, hashtags = self._generate_caption_and_hashtags(selected_topic)
        
        # Step 4: Execution Command Output
        post_time = self._get_post_time()
        
        # Create content object
        content = PostContent(
            day=day,
            selected_topic=selected_topic,
            image_prompt=image_prompt,
            caption=caption,
            hashtags=hashtags,
            post_time=post_time
        )
        
        # Add image path to content if generated
        if image_path:
            content.image_path = image_path
        
        # Step 5: Posting Schedule Logic (handled by scheduler)
        logger.info(f"Post scheduled for {post_time}")
        
        # Step 6: Performance Feedback Ingestion (would be handled by monitoring system)
        if engagement_score:
            self.performance_history.append(engagement_score)
            logger.info(f"Engagement score recorded: {engagement_score}")
        
        # Step 7: Self-Training and Adaptation
        self._update_training_memory(content, engagement_score)
        
        # Prepare execution output
        execution_output = self._prepare_execution_output(content)
        
        logger.info("Daily cycle completed successfully")
        logger.info("=" * 50)
        
        return content, execution_output
    
    def get_status_report(self) -> Dict:
        """Get agent status and performance report"""
        return {
            "last_execution": self.memory["content_history"][-1] if self.memory["content_history"] else None,
            "total_posts": len(self.memory["content_history"]),
            "topic_distribution": self.memory["topic_preferences"],
            "recent_performance": self.performance_history[-7:] if self.performance_history else [],
            "memory_size": len(self.memory["content_history"])
        }


def main():
    """Main execution function"""
    agent = InstagramAIAgent()
    
    # Execute daily cycle
    content, execution_data = agent.execute_daily_cycle()
    
    # Print output in required format
    print("\n" + "="*50)
    print("DAILY EXECUTION OUTPUT")
    print("="*50)
    print(f"DAY: {content.day}")
    print(f"SELECTED_TOPIC: {content.selected_topic}")
    print(f"IMAGE_PROMPT: {content.image_prompt}")
    print(f"CAPTION: {content.caption}")
    print(f"HASHTAGS: {' '.join(content.hashtags)}")
    print(f"POST_TIME: {content.post_time}")
    print(f"TRAINING_UPDATE: Memory updated with new content")
    print("="*50)
    
    # Save execution data
    with open(f"execution_output_{content.day}.json", 'w') as f:
        json.dump(execution_data, f, indent=2)
    
    print(f"\nExecution data saved to execution_output_{content.day}.json")


if __name__ == "__main__":
    main()
