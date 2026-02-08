#!/usr/bin/env python3
"""
Instagram AI Agent Orchestrator

This module orchestrates the entire Instagram content generation pipeline:
1. Text generation (LLM)
2. Image generation
3. Semantic similarity checking
4. Learning and adaptation
5. Content publishing

It integrates all pretrained models into a cohesive automation system.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import our learning engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai_instagram_agent.models.learning.learning_engine import LearningEngine
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

logger = logging.getLogger(__name__)


class InstagramOrchestrator:
    """Main orchestrator for Instagram AI Agent pipeline"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the orchestrator
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = _resolve_config_path(config_path)
        self.config = self._load_config()
        self.config.setdefault("__base_dir", str(self.config_path.parent))
        
        # Initialize learning engine
        self.learning_engine = LearningEngine("learning_memory.json")

        # Initialize model runtime
        self.model_runtime = ModelRuntime(self.config)
        
        # Pipeline state
        self.current_content = {}
        
        logger.info("Instagram orchestrator initialized")
    
    def _load_config(self) -> Dict:
        """Load orchestrator configuration"""
        try:
            with self.config_path.open('r', encoding='utf-8') as f:
                config = json.load(f)
            config["__base_dir"] = str(self.config_path.parent)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found")
            return {}
    
    def generate_text_content(self, topic: str) -> Dict[str, str]:
        """
        Generate text content using LLM (simulated - would integrate with actual model)

        Args:
            topic: Content topic

        Returns:
            Dictionary with caption and hashtags
        """
        logger.info(f"Generating text content for topic: {topic}")

        text_result = self.model_runtime.generate_caption_and_hashtags(topic)

        text_content = {
            "caption": text_result.caption,
            "hashtags": text_result.hashtags,
            "topic": topic
        }

        logger.info(f"Text content generated: {len(text_result.hashtags)} hashtags")
        return text_content

    def generate_image_prompt(self, topic: str) -> str:
        """
        Generate image prompt for Stable Diffusion
        
        Args:
            topic: Content topic
            
        Returns:
            Image generation prompt
        """
        logger.info(f"Generating image prompt for topic: {topic}")
        
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
        
        for key, elements in visual_elements.items():
            if key.lower() in topic.lower():
                base_prompt += f" {elements}"
                break
        
        logger.info(f"Image prompt generated: {base_prompt[:100]}...")
        return base_prompt
    
    def check_semantic_similarity(self, topic: str) -> bool:
        """
        Check if topic is too similar to recent topics.

        Args:
            topic: Topic to check

        Returns:
            True if topic is unique enough, False if too similar
        """
        logger.info(f"Checking semantic similarity for topic: {topic}")

        recent_entries = self.learning_engine.memory.get("engagement_history", [])[-10:]
        recent_topics = [entry.get("topic", "") for entry in recent_entries if entry.get("topic")]

        if self.model_runtime.is_too_similar(topic, recent_topics):
            logger.warning("Topic too similar to recent topics")
            return False

        logger.info("Topic passed similarity check")
        return True

    def execute_pipeline(self, engagement_score: Optional[int] = None) -> Dict:
        """
        Execute the complete Instagram content generation pipeline
        
        Args:
            engagement_score: Optional engagement score from previous post
            
        Returns:
            Complete content package ready for publishing
        """
        logger.info("=" * 60)
        logger.info("Starting Instagram content generation pipeline")
        logger.info("=" * 60)
        
        # Step 1: Get topic (would integrate with LLM for ideation)
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
            "Overcoming procrastination"
        ]
        
        # Select topic based on learning engine recommendations
        recommendations = self.learning_engine.get_recommendations()
        if recommendations["top_topics"]:
            selected_topic = recommendations["top_topics"][0]
        else:
            selected_topic = potential_topics[0]
        
        logger.info(f"Selected topic: {selected_topic}")
        
        # Step 2: Check semantic similarity
        if not self.check_semantic_similarity(selected_topic):
            logger.warning("Topic failed similarity check, selecting alternative")
            selected_topic = potential_topics[1]  # Fallback
        
        # Step 3: Generate text content
        text_content = self.generate_text_content(selected_topic)
        
        # Step 4: Generate image prompt
        image_prompt = self.generate_image_prompt(selected_topic)

        # Step 5: Generate image if enabled
        image_path = None
        if self.model_runtime.should_generate_images():
            image_path = self.model_runtime.generate_image(image_prompt)
        
        # Step 6: Package content
        content_package = {
            "day": datetime.datetime.now().strftime("%Y-%m-%d"),
            "topic": selected_topic,
            "image_prompt": image_prompt,
            "image_path": image_path,
            "caption": text_content["caption"],
            "hashtags": text_content["hashtags"],
            "post_time": self.config.get("posting_schedule", {}).get("daily_time", "14:00"),
            "status": "ready_for_publishing"
        }
        
        # Step 7: Update learning if engagement score provided
        if engagement_score is not None:
            self.learning_engine.update_performance(
                topic=selected_topic,
                hashtags=text_content["hashtags"],
                caption=text_content["caption"],
                engagement_score=engagement_score
            )
        
        logger.info("Pipeline execution completed successfully")
        logger.info("=" * 60)
        
        return content_package
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status and recommendations"""
        learning_summary = self.learning_engine.get_performance_summary()
        recommendations = self.learning_engine.get_recommendations()
        
        return {
            "pipeline_status": "ready",
            "learning_summary": learning_summary,
            "content_recommendations": recommendations,
            "next_execution_time": self.config.get("posting_schedule", {}).get("daily_time", "14:00"),
            "total_content_generated": len(self.learning_engine.memory["engagement_history"])
        }
    
    def simulate_publishing(self, content_package: Dict) -> bool:
        """
        Publish content using configured publisher or simulate publishing.

        Args:
            content_package: Content to publish

        Returns:
            True if publishing successful
        """
        logger.info(f"Simulating publishing for topic: {content_package['topic']}")

        publisher_config = self.config.get("publisher", {})
        if publisher_config.get("enabled", False):
            try:
                from instagram_simple_post import publish_image
            except Exception as exc:
                logger.error(f"Publisher import failed: {exc}")
                return False

            image_path = content_package.get("image_path")
            if not image_path:
                logger.error("Publishing enabled but no image_path was generated")
                return False

            caption = f"{content_package['caption']}\n\n{' '.join(content_package['hashtags'])}"
            try:
                publish_image(image_path, caption)
                logger.info("Content published successfully")
                return True
            except Exception as exc:
                logger.error(f"Publishing failed: {exc}")
                return False

        logger.info("Content published successfully (simulation)")
        return True


def main():
    """Test the orchestrator"""
    orchestrator = InstagramOrchestrator()
    
    # Execute pipeline
    content = orchestrator.execute_pipeline()
    
    print("Generated Content:")
    print(f"Topic: {content['topic']}")
    print(f"Image Prompt: {content['image_prompt'][:100]}...")
    print(f"Caption Length: {len(content['caption'].split())} words")
    print(f"Hashtags: {len(content['hashtags'])}")
    
    # Get status
    status = orchestrator.get_pipeline_status()
    print("\nPipeline Status:")
    print(f"Learning Summary: {status['learning_summary']}")
    print(f"Recommendations: {status['content_recommendations']}")


if __name__ == "__main__":
    main()
