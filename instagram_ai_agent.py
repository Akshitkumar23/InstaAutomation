#!/usr/bin/env python3
"""
Instagram AI Agent - Autonomous Content Publishing System

This module implements a fully automated, end-to-end Instagram content publishing system
that plans, generates, executes, evaluates, and improves Instagram posts without human intervention.
"""
# pyre-ignore-all-errors

import os
import json
import logging
import datetime
import hashlib
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from ai_instagram_agent.model_runtime import ModelRuntime  # type: ignore

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
    image_path: Optional[str] = None
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
            # Productivity & Time Management (15 topics)
            "Productivity hacks for remote workers",
            "Morning routine optimization",
            "Digital minimalism principles",
            "Time management techniques",
            "Focus and concentration methods",
            "Deep work strategies",
            "Pomodoro technique mastery",
            "Inbox zero methodology",
            "Task batching benefits",
            "Energy management vs time management",
            "Eliminating time wasters",
            "Priority matrix framework",
            "Weekly planning rituals",
            "Automation for productivity",
            "Single-tasking advantages",
            
            # Personal Growth & Mindset (15 topics)
            "Personal growth mindset",
            "Building positive habits",
            "Overcoming procrastination",
            "Decision making frameworks",
            "Growth mindset vs fixed mindset",
            "Embracing failure as learning",
            "Self-awareness practices",
            "Limiting beliefs identification",
            "Confidence building techniques",
            "Resilience development",
            "Emotional intelligence skills",
            "Self-discipline strategies",
            "Gratitude practice benefits",
            "Journaling for clarity",
            "Identity-based habits",
            
            # Health & Wellness (10 topics)
            "Mindfulness in daily life",
            "Digital detox benefits",
            "Sleep optimization tips",
            "Stress management techniques",
            "Meditation for beginners",
            "Exercise habit formation",
            "Nutrition fundamentals",
            "Mental health awareness",
            "Work-life balance tips",
            "Burnout prevention",
            
            # Learning & Skills (10 topics)
            "Creative problem solving",
            "Learning new skills efficiently",
            "Speed reading techniques",
            "Memory improvement methods",
            "Critical thinking development",
            "Active learning strategies",
            "Note-taking systems",
            "Skill stacking approach",
            "Learning from failures",
            "Continuous improvement mindset",
            
            # Goals & Achievement (10 topics)
            "Goal setting strategies",
            "SMART goals framework",
            "Vision board creation",
            "Milestone tracking",
            "Accountability systems",
            "Progress measurement",
            "Celebrating small wins",
            "Long-term planning",
            "Quarterly reviews",
            "Annual goal setting",
            
            # Communication & Relationships (5 topics)
            "Effective communication skills",
            "Active listening techniques",
            "Networking strategies",
            "Boundary setting",
            "Conflict resolution",
            
            # Finance & Career (5 topics)
            "Financial literacy basics",
            "Career development planning",
            "Side hustle ideas",
            "Passive income streams",
            "Investment fundamentals"
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
        Generate SHORT image prompt for faster generation
        """
        logger.info("Step 2: Generating image prompt")
        
        # Extremely detailed mountain scenes for hyper-realistic generation
        nature_scenes = [
            "A breathtaking panoramic view of the snow-capped Annapurna mountain range in the Himalayas during golden hour, the jagged peaks are illuminated in vibrant orange and pink hues while the valley below remains in deep cool blue shadow, highly detailed rock textures and snow patterns, swirling mist around the base adds mystery, shot on a high-resolution Sony A7R IV with a 16mm wide-angle lens to capture the vastness, cinematic lighting, 8k resolution, award-winning national geographic style photography",
            
            "A serene alpine lake acting as a perfect mirror reflecting the towering granite peaks of the Dolomites, crucial details include the crystal clear water showing pebbles at the bottom in the foreground, scattered wildflowers adding pops of yellow and purple color near the shore, soft morning light filtering through a thin layer of fog, hyper-realistic water reflections, calm atmosphere, shot at f/8 aperture for deep depth of field, 8k masterpiece",
            
            "An adventurous winding road cutting through a lush green mountain pass in the Swiss Alps, surrounded by towering pine trees and distant waterfalls cascading down sheer cliffs, a dramatic storm cloud formation is gathering in the sky creating a moody and atmospheric lighting condition, the asphalt road is wet from recent rain reflecting the grey sky, high contrast, cinematic composition, ultra-detailed foliage and rock surfaces",
            
            "A majestic solitary mountain cabin buried under deep pristine white show in a dense forest at twilight, warm inviting golden light spills out from the cabin windows creating a cozy contrast against the cold blue winter landscape, smoke gently rising from the stone chimney, stars beginning to appear in the dark purple sky above the tree line, magical winter wonderland atmosphere, highly detailed snow texture, photorealistic rendering",
            
            "A dramatic low-angle shot of a rugged mountain ridge silhouette against the Milky Way galaxy in the night sky, thousands of bright stars and the galactic core are clearly visible above the dark sharp outlines of the peaks, a faint glow of a campfire is visible in the distance add a human element, long exposure astrophotography style, noise-free high ISO, spiritual and awe-inspiring mood, 8k resolution"
        ]
        
        # Pick a random scene
        import random
        scene = random.choice(nature_scenes)
        
        # Enhanced realism prompt suffix
        suffix = "8k resolution, photorealistic, cinematic lighting, highly detailed, shot on 35mm lens, sharp focus, masterpiece, trending on artstation, ultra-realistic textures"
        prompt = f"{scene}, {suffix}"
        
        logger.info(f"Generated nature-focused image prompt: {prompt}")
        return prompt
    
    
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
        
        # Generate caption with engaging templates
        caption_templates = [
            # Hook + Value + CTA format
            f"* {topic} made simple:\n\n"
            "1) Identify the bottleneck\n"
            "2) Build a tiny routine\n"
            "3) Review weekly\n\n"
            "Share this with someone who needs it.",
            
            # Question Hook + Breakdown format
            f"* Quick breakdown: {topic}\n\n"
            "Why it matters:\n"
            "- Clarity\n"
            "- Consistency\n"
            "- Compounding results\n\n"
            "Which point will you try first?",
            
            # Problem-Solution format
            f"* Struggling with {topic}?\n\n"
            "Here's what actually works:\n\n"
            "→ Start with one small step\n"
            "→ Keep it simple and repeatable\n"
            "→ Track progress for 7 days\n\n"
            "Save this post if it helps.",
            
            # Myth-Buster format
            f"* The truth about {topic}:\n\n"
            "Most people overcomplicate it.\n\n"
            "Reality:\n"
            "• Focus on fundamentals\n"
            "• Build systems, not goals\n"
            "• Measure what matters\n\n"
            "Tag someone who needs this.",
            
            # Story format
            f"* {topic}! Here are 3 practical ideas you can try today:\n\n"
            "1. Start with one small step\n"
            "2. Keep it simple and repeatable\n"
            "3. Track progress for 7 days\n\n"
            "Save this post if it helps.",
            
            # Listicle format
            f"* 3 things I wish I knew about {topic}:\n\n"
            "1. Consistency beats intensity\n"
            "2. Systems beat motivation\n"
            "3. Progress beats perfection\n\n"
            "Which one resonates with you?",
            
            # Challenge format
            f"* 7-day {topic} challenge:\n\n"
            "Day 1-2: Learn the basics\n"
            "Day 3-5: Practice daily\n"
            "Day 6-7: Review & adjust\n\n"
            "Who's in? Comment 'I'm in' below!",
            
            # Mistake format
            f"* Common mistakes with {topic}:\n\n"
            "❌ Trying to do everything at once\n"
            "✅ Focus on one thing at a time\n\n"
            "❌ Waiting for motivation\n"
            "✅ Build systems instead\n\n"
            "Save this for later.",
            
            # Before/After format
            f"* How {topic} changed my perspective:\n\n"
            "Before: Overwhelmed and stuck\n"
            "After: Clear and consistent\n\n"
            "The difference?\n"
            "→ Simple daily actions\n"
            "→ Tracking progress\n"
            "→ Staying patient\n\n"
            "Your turn. Start today.",
            
            # Quick Win format
            f"* Quick win for {topic}:\n\n"
            "Instead of planning for hours,\n"
            "Just do this:\n\n"
            "1. Pick ONE thing\n"
            "2. Do it for 5 minutes\n"
            "3. Repeat tomorrow\n\n"
            "Simple wins compound.",
        ]
        
        caption = random.choice(caption_templates)
        
        # Ensure caption length constraint (trim by words, not characters)
        max_words = self.config["content_preferences"]["max_caption_length"]
        words = caption.split()
        if len(words) > max_words:
            caption = " ".join(words[:max_words]) # type: ignore
        
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
        self.memory["content_history"].append(asdict(content)) # type: ignore
        
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
            recent_scores = self.performance_history[-7:] # type: ignore
            if all(score < self.config["performance_thresholds"]["min_engagement_score"] 
                   for score in recent_scores[-3:]):
                logger.warning("Engagement decline detected - entering exploration mode")
                # Implementation would include experimenting with new patterns
        
        self._save_memory()
        logger.info("Training memory updated")
    
    def execute_daily_cycle(self, engagement_score: Optional[int] = None) -> Tuple[PostContent, Dict]:
        """
        Execute one complete daily cycle of the Instagram AI Agent
        
        Args:
            engagement_score: Optional engagement score from previous post
            
        Returns:
            Tuple[PostContent, Dict]: Generated content and execution output for today
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
            "recent_performance": self.performance_history[-7:] if self.performance_history else [], # type: ignore
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
