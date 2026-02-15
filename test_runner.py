import sys
import os

# Add current directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instagram_ai_agent import InstagramAIAgent  # type: ignore

def main():
    print("Starting test run with config_test.json...")
    
    # Initialize agent with test config
    agent = InstagramAIAgent(config_path="config_test.json")
    
    # Execute daily cycle
    try:
        content, execution_data = agent.execute_daily_cycle()
        
        print("\n" + "="*50)
        print("TEST EXECUTION SUCCESSFUL")
        print("="*50)
        print(f"Topic: {content.selected_topic}")
        print(f"Image Prompt: {content.image_prompt}")
        print(f"Caption: {content.caption}")
        print(f"Image Path: {execution_data.get('image_generation', {}).get('prompt', 'N/A')}") 
        # Note: image path isn't directly in execution_data['image_generation'], checking what is there.
        # execution_data['image_generation'] has 'prompt', 'dimensions', 'style'.
        # The content object might have image_path if the code added it.
        if hasattr(content, 'image_path'):
            print(f"Generated Image: {content.image_path}")
        
    except Exception as e:
        print(f"TEST EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
