# Instagram AI Agent

Compatibility Notes
===================
- The automation pipeline now reads `config.json` for model providers and publishing settings.
- See `README.md` for provider configuration details.


A fully automated, end-to-end Instagram content publishing system that plans, generates, executes, evaluates, and improves Instagram posts without human intervention.

## 🤖 System Overview

The Instagram AI Agent is an autonomous system that performs the following core responsibilities:

1. **Content Strategy Decision** - Selects unique topics daily with 30-day rotation
2. **Image Prompt Generation** - Creates detailed prompts for 1080x1080 Instagram posts
3. **Caption & Hashtag Generation** - Generates engaging captions with optimized hashtags
4. **Execution Output** - Prepares structured data for downstream modules
5. **Performance Feedback** - Tracks engagement metrics and calculates scores
6. **Self-Training** - Adapts and improves based on performance data
7. **Scheduled Publishing** - Manages daily posting at optimal times

## 📁 Project Structure

```
instagram_ai_agent/
├── README.md                    # Main project documentation
├── INSTAGRAM_AI_AGENT_README.md # This file
├── instagram_ai_agent.py        # Core agent implementation
├── scheduler.py                 # Automated scheduling system
├── example_usage.py             # Usage examples and demos
├── config.json                  # Configuration settings
├── requirements.txt             # Python dependencies
├── config.py                    # Original Instagram posting library
├── instagram_simple_post.py     # Original posting function
└── requirements.txt             # Original dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install core dependencies only
pip install python-dotenv requests filestack-python
```

### 2. Configuration

Edit `config.json` with your credentials:

```json
{
  "api_credentials": {
    "instagram_access_token": "your_instagram_token",
    "instagram_user_id": "your_user_id", 
    "filestack_api_key": "your_filestack_key"
  },
  "posting_schedule": {
    "daily_time": "14:00",
    "timezone": "Asia/Calcutta"
  }
}
```

### 3. Basic Usage

```python
from instagram_ai_agent import InstagramAIAgent

# Initialize agent
agent = InstagramAIAgent()

# Execute daily cycle
content, execution_data = agent.execute_daily_cycle()

# View results
print(f"Topic: {content.selected_topic}")
print(f"Image prompt: {content.image_prompt}")
print(f"Caption: {content.caption}")
print(f"Hashtags: {' '.join(content.hashtags)}")
```

### 4. Automated Scheduling

```bash
# Run once
python scheduler.py --mode once

# Start daily scheduler
python scheduler.py --mode daily

# Start as background daemon
python scheduler.py --mode daemon

# Run system tests
python scheduler.py --mode test

# Check status
python scheduler.py --mode status
```

## 🔧 Configuration Options

### API Credentials
- `instagram_access_token`: Instagram Graph API access token
- `instagram_user_id`: Your Instagram user ID
- `filestack_api_key`: Filestack API key for media uploads

### Posting Schedule
- `daily_time`: Optimal posting time (HH:MM format)
- `timezone`: Timezone for scheduling
- `enabled`: Enable/disable automated posting

### Content Preferences
- `max_caption_length`: Maximum caption length in words
- `min_hashtags` / `max_hashtags`: Hashtag count range
- `preferred_emojis`: List of preferred emojis
- `banned_hashtags`: List of hashtags to avoid

### Performance Thresholds
- `engagement_decline_threshold`: Days of decline before exploration mode
- `min_engagement_score`: Minimum acceptable engagement score

## 📊 Output Format

Each daily execution produces structured output:

```
DAY: 2024-01-15
SELECTED_TOPIC: Productivity hacks for remote workers
IMAGE_PROMPT: Create a 1080x1080 Instagram feed post image about 'Productivity hacks for remote workers'...
CAPTION: 🚀 Productivity hacks for remote workers! Here's how you can implement this in your daily life...
HASHTAGS: #productivity #remotework #workfromhome #productivitytips #remoteworklife #digitalnomad #worklifebalance #productivityhacks #remoteworktips #productivitytools #remoteworksetup #productivitycoach #remoteworker #productivityapp #remoteworkspace
POST_TIME: 14:00
TRAINING_UPDATE: Memory updated with new content
```

## 🔄 Daily Execution Cycle

The agent follows a strict 7-step process:

1. **Content Strategy**: Select unique topic (30-day rotation)
2. **Image Generation**: Create detailed image prompt
3. **Caption Creation**: Generate engaging caption with CTA
4. **Hashtag Selection**: Choose 15-25 optimized hashtags
5. **Execution Prep**: Structure data for downstream modules
6. **Performance Tracking**: Record engagement metrics
7. **Self-Optimization**: Update memory and adapt strategies

## 📈 Performance Monitoring

The agent tracks:
- Engagement scores (likes, comments, shares, saves)
- Topic performance over time
- Hashtag effectiveness
- Content type preferences
- Posting time optimization

## 🎯 Content Strategy

### Topic Rotation
- **Educational**: Learning, skills, knowledge-based content
- **Motivational**: Inspiration, encouragement, mindset
- **System-based**: Processes, methods, frameworks
- **Habit-based**: Routines, daily practices, consistency
- **Insight-based**: Reflections, observations, wisdom

### Topic Examples
- Productivity hacks for remote workers
- Morning routine optimization
- Digital minimalism principles
- Goal setting strategies
- Time management techniques

## 🔍 Memory Management

The agent maintains persistent memory in `agent_memory.json`:
- Content history (last 30 days)
- Topic preferences and rotation
- Hashtag performance tracking
- Engagement score history
- Performance optimization patterns

## 🛠️ Development

### Running Tests

```bash
# Run example demonstrations
python example_usage.py

# Run system tests
python scheduler.py --mode test

# Check agent status
python scheduler.py --mode status
```

### Custom Configuration

Create custom config files for different environments:

```bash
# Development config
python scheduler.py --config dev_config.json --mode once

# Production config  
python scheduler.py --config prod_config.json --mode daemon
```

### Extending Functionality

The agent is designed for extensibility:

1. **Add new content types** in `_classify_content_type()`
2. **Customize caption templates** in `_generate_caption_and_hashtags()`
3. **Add visual elements** in `_generate_image_prompt()`
4. **Implement new performance metrics** in `_update_training_memory()`

## 🚨 Important Notes

- **Autonomous Operation**: The agent operates without human intervention
- **Topic Uniqueness**: Ensures no topic repetition within 30 days
- **Performance Optimization**: Automatically adapts based on engagement
- **Error Handling**: Comprehensive logging and error recovery
- **Memory Persistence**: All data persists across executions

## 🔒 Security

- Store API credentials securely
- Use environment variables for sensitive data
- Regularly rotate API tokens
- Monitor access logs for security

## 📞 Support

For issues, questions, or contributions:

1. Check the logs in `instagram_agent.log` and `scheduler.log`
2. Run `python scheduler.py --mode test` to verify setup
3. Review configuration in `config.json`
4. Check agent status with `python scheduler.py --mode status`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This system is designed for automated Instagram content publishing. Ensure compliance with Instagram's terms of service and community guidelines.