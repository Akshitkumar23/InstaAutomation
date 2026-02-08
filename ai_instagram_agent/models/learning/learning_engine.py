#!/usr/bin/env python3
"""
Learning Engine for Instagram AI Agent

This module implements reward-based learning logic to optimize content performance
without using deep learning. It tracks engagement scores and adjusts content strategy.
"""

import json
import logging
import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningEngine:
    """Reward-based learning engine for content optimization"""
    
    def __init__(self, memory_path: str = "learning_memory.json"):
        """
        Initialize the learning engine
        
        Args:
            memory_path: Path to learning memory file
        """
        self.memory_path = Path(memory_path)
        self.memory = self._load_memory()
        
        # Learning parameters
        self.exploration_rate = 0.1  # 10% exploration
        self.performance_window = 7  # Track last 7 posts
        self.decline_threshold = 3   # Days of decline before exploration
        
        logger.info("Learning engine initialized")
    
    def _load_memory(self) -> Dict:
        """Load learning memory from file"""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading learning memory: {e}")
        
        return {
            "topic_performance": {},
            "hashtag_performance": {},
            "caption_patterns": {},
            "engagement_history": [],
            "exploration_mode": False,
            "last_exploration": None
        }
    
    def _save_memory(self):
        """Save learning memory to file"""
        try:
            with open(self.memory_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning memory: {e}")
    
    def update_performance(self, topic: str, hashtags: List[str], 
                          caption: str, engagement_score: int):
        """
        Update performance metrics for learning
        
        Args:
            topic: Content topic
            hashtags: List of hashtags used
            caption: Caption text
            engagement_score: Engagement score for this post
        """
        # Record engagement history
        self.memory["engagement_history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "topic": topic,
            "hashtags": hashtags,
            "caption": caption,
            "engagement_score": engagement_score
        })
        
        # Keep only recent history
        if len(self.memory["engagement_history"]) > 100:
            self.memory["engagement_history"] = self.memory["engagement_history"][-100:]
        
        # Update topic performance
        if topic not in self.memory["topic_performance"]:
            self.memory["topic_performance"][topic] = []
        self.memory["topic_performance"][topic].append(engagement_score)
        
        # Update hashtag performance
        for hashtag in hashtags:
            if hashtag not in self.memory["hashtag_performance"]:
                self.memory["hashtag_performance"][hashtag] = []
            self.memory["hashtag_performance"][hashtag].append(engagement_score)
        
        # Update caption patterns
        caption_length = len(caption.split())
        if caption_length not in self.memory["caption_patterns"]:
            self.memory["caption_patterns"][caption_length] = []
        self.memory["caption_patterns"][caption_length].append(engagement_score)
        
        # Check for performance decline
        self._check_performance_decline()
        
        # Save memory
        self._save_memory()
        
        logger.info(f"Performance updated: topic={topic}, score={engagement_score}")
    
    def _check_performance_decline(self):
        """Check if performance has declined and enter exploration mode"""
        if len(self.memory["engagement_history"]) < self.performance_window:
            return
        
        # Get recent engagement scores
        recent_scores = [entry["engagement_score"] for entry in 
                        self.memory["engagement_history"][-self.performance_window:]]
        
        # Check for decline
        if len(recent_scores) >= self.decline_threshold:
            recent_avg = sum(recent_scores[-self.decline_threshold:]) / self.decline_threshold
            older_avg = sum(recent_scores[:-self.decline_threshold]) / (len(recent_scores) - self.decline_threshold)
            
            if recent_avg < older_avg * 0.8:  # 20% decline
                self.memory["exploration_mode"] = True
                self.memory["last_exploration"] = datetime.datetime.now().isoformat()
                logger.warning(f"Performance decline detected. Entering exploration mode.")
    
    def get_topic_score(self, topic: str) -> float:
        """
        Get performance score for a topic
        
        Args:
            topic: Topic to evaluate
            
        Returns:
            Average engagement score for the topic
        """
        if topic not in self.memory["topic_performance"]:
            return 50.0  # Default score
        
        scores = self.memory["topic_performance"][topic]
        return sum(scores) / len(scores)
    
    def get_hashtag_score(self, hashtag: str) -> float:
        """
        Get performance score for a hashtag
        
        Args:
            hashtag: Hashtag to evaluate
            
        Returns:
            Average engagement score for the hashtag
        """
        if hashtag not in self.memory["hashtag_performance"]:
            return 50.0  # Default score
        
        scores = self.memory["hashtag_performance"][hashtag]
        return sum(scores) / len(scores)
    
    def get_optimal_caption_length(self) -> int:
        """
        Get optimal caption length based on performance
        
        Returns:
            Recommended caption length in words
        """
        if not self.memory["caption_patterns"]:
            return 50  # Default length
        
        best_length = 0
        best_score = 0
        
        for length, scores in self.memory["caption_patterns"].items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_length = length
        
        return best_length
    
    def should_explore(self) -> bool:
        """
        Check if system should explore new content strategies
        
        Returns:
            True if exploration mode is active
        """
        return self.memory["exploration_mode"]
    
    def reset_exploration(self):
        """Reset exploration mode after trying new strategies"""
        self.memory["exploration_mode"] = False
        logger.info("Exploration mode reset")
    
    def get_recommendations(self) -> Dict[str, any]:
        """
        Get content recommendations based on learning
        
        Returns:
            Dictionary with recommendations for topics, hashtags, and caption length
        """
        # Get top performing topics
        topic_scores = {topic: self.get_topic_score(topic) 
                       for topic in self.memory["topic_performance"]}
        top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Get top performing hashtags
        hashtag_scores = {hashtag: self.get_hashtag_score(hashtag) 
                         for hashtag in self.memory["hashtag_performance"]}
        top_hashtags = sorted(hashtag_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get optimal caption length
        optimal_length = self.get_optimal_caption_length()
        
        return {
            "top_topics": [topic for topic, score in top_topics],
            "top_hashtags": [hashtag for hashtag, score in top_hashtags],
            "optimal_caption_length": optimal_length,
            "exploration_mode": self.should_explore(),
            "total_posts_analyzed": len(self.memory["engagement_history"])
        }
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Get overall performance summary"""
        if not self.memory["engagement_history"]:
            return {"status": "No data available"}
        
        scores = [entry["engagement_score"] for entry in self.memory["engagement_history"]]
        
        return {
            "total_posts": len(self.memory["engagement_history"]),
            "average_engagement": sum(scores) / len(scores),
            "best_engagement": max(scores),
            "worst_engagement": min(scores),
            "topics_tracked": len(self.memory["topic_performance"]),
            "hashtags_tracked": len(self.memory["hashtag_performance"]),
            "exploration_mode": self.should_explore()
        }


def main():
    """Test the learning engine"""
    engine = LearningEngine()
    
    # Simulate some performance data
    test_data = [
        ("Productivity tips", ["#productivity", "#tips"], "Short caption", 100),
        ("Time management", ["#timemanagement", "#productivity"], "Medium length caption with more details", 150),
        ("Morning routine", ["#morning", "#routine"], "Very long caption with lots of details and information", 80),
    ]
    
    for topic, hashtags, caption, score in test_data:
        engine.update_performance(topic, hashtags, caption, score)
    
    # Get recommendations
    recommendations = engine.get_recommendations()
    print("Recommendations:", recommendations)
    
    # Get performance summary
    summary = engine.get_performance_summary()
    print("Performance Summary:", summary)


if __name__ == "__main__":
    main()