#!/usr/bin/env python3
"""
Example usage script for Instagram AI Agent

This script demonstrates how to use the Instagram AI Agent for various scenarios.
"""

import json
import time
from datetime import datetime, timedelta
from instagram_ai_agent import InstagramAIAgent, PostContent


def demo_basic_usage():
    """Demonstrate basic daily cycle execution"""
    print("=== Basic Usage Demo ===")
    
    # Initialize agent
    agent = InstagramAIAgent()
    
    # Execute daily cycle
    content, execution_data = agent.execute_daily_cycle()
    
    # Display results
    print(f"Generated content for {content.day}")
    print(f"Topic: {content.selected_topic}")
    print(f"Image prompt: {content.image_prompt[:100]}...")
    print(f"Caption length: {len(content.caption.split())} words")
    print(f"Hashtags: {len(content.hashtags)}")
    print(f"Scheduled for: {content.post_time}")
    
    return content, execution_data


def demo_with_engagement_feedback():
    """Demonstrate usage with engagement score feedback"""
    print("\n=== Engagement Feedback Demo ===")
    
    agent = InstagramAIAgent()
    
    # Simulate previous engagement scores
    previous_scores = [120, 95, 150, 80, 200, 180, 160]
    agent.performance_history = previous_scores
    
    # Execute with engagement score
    content, execution_data = agent.execute_daily_cycle(engagement_score=140)
    
    print(f"Content generated with engagement feedback")
    print(f"Recent performance: {previous_scores}")
    print(f"New engagement score: 140")
    
    return content, execution_data


def demo_status_report():
    """Demonstrate status reporting"""
    print("\n=== Status Report Demo ===")
    
    agent = InstagramAIAgent()
    
    # Get status report
    status = agent.get_status_report()
    
    print("Agent Status Report:")
    print(f"Total posts generated: {status['total_posts']}")
    print(f"Topic distribution: {status['topic_distribution']}")
    print(f"Recent performance: {status['recent_performance']}")
    print(f"Memory size: {status['memory_size']} entries")
    
    if status['last_execution']:
        last = status['last_execution']
        print(f"Last execution: {last['day']} - {last['selected_topic']}")


def demo_multiple_days():
    """Demonstrate running for multiple days"""
    print("\n=== Multiple Days Demo ===")
    
    agent = InstagramAIAgent()
    
    # Simulate running for 5 days
    for i in range(5):
        print(f"\nDay {i+1}:")
        
        # Generate random engagement score for demonstration
        engagement_score = 50 + (i * 20) + (i % 3) * 10
        
        content, execution_data = agent.execute_daily_cycle(engagement_score)
        
        print(f"  Topic: {content.selected_topic}")
        print(f"  Engagement: {engagement_score}")
        print(f"  Content type: {agent._classify_content_type(content.selected_topic)}")
        
        # Small delay to simulate real time
        time.sleep(0.1)


def demo_custom_config():
    """Demonstrate using custom configuration"""
    print("\n=== Custom Configuration Demo ===")
    
    # Create custom config
    custom_config = {
        "api_credentials": {
            "instagram_access_token": "your_token_here",
            "instagram_user_id": "your_user_id",
            "filestack_api_key": "your_filestack_key"
        },
        "posting_schedule": {
            "daily_time": "18:00",  # 6 PM
            "timezone": "Asia/Calcutta"
        },
        "content_preferences": {
            "max_caption_length": 200,
            "min_hashtags": 10,
            "max_hashtags": 20,
            "preferred_emojis": ["🎯", "💡", "🚀"],
            "banned_hashtags": ["spam", "fake"]
        }
    }
    
    # Save custom config
    with open("custom_config.json", 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    # Initialize agent with custom config
    agent = InstagramAIAgent("custom_config.json")
    
    content, execution_data = agent.execute_daily_cycle()
    
    print(f"Using custom configuration")
    print(f"Posting time: {agent.config['posting_schedule']['daily_time']}")
    print(f"Max caption length: {agent.config['content_preferences']['max_caption_length']}")
    print(f"Selected topic: {content.selected_topic}")
    
    # Clean up
    import os
    if os.path.exists("custom_config.json"):
        os.remove("custom_config.json")


def demo_memory_management():
    """Demonstrate memory management features"""
    print("\n=== Memory Management Demo ===")
    
    agent = InstagramAIAgent()
    
    # Check initial memory
    print(f"Initial memory entries: {len(agent.memory['content_history'])}")
    
    # Simulate some content generation
    for i in range(3):
        content, _ = agent.execute_daily_cycle(engagement_score=100 + i*20)
    
    print(f"After 3 cycles: {len(agent.memory['content_history'])} entries")
    
    # Check topic rotation
    topics = [entry['selected_topic'] for entry in agent.memory['content_history']]
    print(f"Recent topics: {[t[:30] + '...' if len(t) > 30 else t for t in topics]}")
    
    # Check if topic repetition is avoided
    agent2 = InstagramAIAgent()
    new_topic = agent2._select_content_topic()
    print(f"New topic selection respects rotation: {new_topic}")


def main():
    """Run all demonstration functions"""
    print("Instagram AI Agent - Example Usage")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demo_basic_usage()
        demo_with_engagement_feedback()
        demo_status_report()
        demo_multiple_days()
        demo_custom_config()
        demo_memory_management()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("Check the generated files:")
        print("- instagram_agent.log (execution logs)")
        print("- agent_memory.json (agent memory)")
        print("- execution_output_*.json (daily outputs)")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()