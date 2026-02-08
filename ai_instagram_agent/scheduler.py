#!/usr/bin/env python3
"""
Instagram AI Agent Scheduler

This module provides scheduling for the Instagram AI Agent orchestrator.
It can run in different modes:
- Single execution
- Daily scheduled execution
- Background daemon mode
"""

import os
import sys
import time
import json
import logging
import schedule
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import InstagramOrchestrator

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


class InstagramScheduler:
    """Scheduler for Instagram AI Agent orchestrator"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the scheduler
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = _resolve_config_path(config_path)
        self.config = self._load_config()
        self.orchestrator = InstagramOrchestrator(str(self.config_path))
        self.is_running = False
        self.last_execution_time = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Instagram AI Agent Scheduler initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load scheduler configuration"""
        try:
            with self.config_path.open('r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found")
            return {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get("monitoring", {}).get("log_level", "INFO")
        log_file = "scheduler.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _get_scheduled_time(self) -> str:
        """Get the scheduled posting time"""
        return self.config.get("posting_schedule", {}).get("daily_time", "14:00")
    
    def _should_run_today(self) -> bool:
        """Check if agent should run today"""
        schedule_config = self.config.get("posting_schedule", {})
        
        if not schedule_config.get("enabled", True):
            return False
        
        # Check if today is a holiday or maintenance day (future enhancement)
        # For now, always run daily
        return True
    
    def _calculate_next_run(self) -> datetime:
        """Calculate next scheduled run time"""
        scheduled_time = self._get_scheduled_time()
        hour, minute = map(int, scheduled_time.split(':'))
        
        now = datetime.now()
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If the time has already passed today, schedule for tomorrow
        if next_run <= now:
            next_run += timedelta(days=1)
        
        return next_run
    
    def _run_orchestrator_once(self, engagement_score: Optional[int] = None):
        """Execute one cycle of the Instagram orchestrator"""
        try:
            logger.info("Starting scheduled orchestrator execution")
            
            # Execute pipeline
            content_package = self.orchestrator.execute_pipeline(engagement_score)
            
            # Simulate publishing
            self.orchestrator.simulate_publishing(content_package)
            
            # Update execution time
            self.last_execution_time = datetime.now()
            
            logger.info(f"Orchestrator execution completed successfully")
            logger.info(f"Topic: {content_package['topic']}")
            logger.info(f"Scheduled for: {content_package['post_time']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during orchestrator execution: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_scheduled_job(self):
        """Scheduled job function for the scheduler"""
        if not self._should_run_today():
            logger.info("Skipping execution today (schedule disabled or maintenance)")
            return
        
        success = self._run_orchestrator_once()
        
        if success:
            logger.info("Scheduled execution completed successfully")
        else:
            logger.error("Scheduled execution failed")
    
    def run_once(self, engagement_score: Optional[int] = None):
        """Run the orchestrator once immediately"""
        logger.info("Running orchestrator once (manual execution)")
        return self._run_orchestrator_once(engagement_score)
    
    def start_daily_scheduler(self):
        """Start the daily scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        scheduled_time = self._get_scheduled_time()
        
        # Schedule the job
        schedule.every().day.at(scheduled_time).do(self._run_scheduled_job)
        
        self.is_running = True
        logger.info(f"Daily scheduler started - running at {scheduled_time} daily")
        
        # Calculate and display next run time
        next_run = self._calculate_next_run()
        logger.info(f"Next execution scheduled for: {next_run}")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self.stop()
    
    def start_background_daemon(self):
        """Start the scheduler as a background daemon"""
        def run_scheduler():
            self.start_daily_scheduler()
        
        # Start scheduler in a separate thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Background daemon started")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(3600)  # Sleep for 1 hour
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
            self.stop()
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        schedule.clear()
        logger.info("Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        next_run = self._calculate_next_run() if self.is_running else None
        
        return {
            "is_running": self.is_running,
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "next_execution": next_run.isoformat() if next_run else None,
            "scheduled_time": self._get_scheduled_time(),
            "orchestrator_status": self.orchestrator.get_pipeline_status()
        }
    
    def run_test_cycle(self):
        """Run a test cycle to verify everything works"""
        logger.info("Running test cycle...")
        
        # Test orchestrator initialization
        try:
            test_orchestrator = InstagramOrchestrator(self.config_path)
            logger.info("✓ Orchestrator initialization successful")
        except Exception as e:
            logger.error(f"✗ Orchestrator initialization failed: {e}")
            return False
        
        # Test content generation
        try:
            content_package = test_orchestrator.execute_pipeline()
            logger.info("✓ Content generation successful")
            logger.info(f"  Topic: {content_package['topic']}")
            logger.info(f"  Hashtags: {len(content_package['hashtags'])}")
        except Exception as e:
            logger.error(f"✗ Content generation failed: {e}")
            return False
        
        # Test publishing simulation
        try:
            test_orchestrator.simulate_publishing(content_package)
            logger.info("✓ Publishing simulation successful")
        except Exception as e:
            logger.error(f"✗ Publishing simulation failed: {e}")
            return False
        
        logger.info("✓ All tests passed")
        return True


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Instagram AI Agent Scheduler")
    parser.add_argument(
        "--mode", 
        choices=["once", "daily", "daemon", "test", "status"],
        default="once",
        help="Execution mode"
    )
    parser.add_argument(
        "--config", 
        default="config.json",
        help="Configuration file path"
    )
    parser.add_argument(
        "--engagement", 
        type=int,
        help="Engagement score for this execution"
    )
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = InstagramScheduler(args.config)
    
    if args.mode == "once":
        # Run once
        success = scheduler.run_once(args.engagement)
        sys.exit(0 if success else 1)
        
    elif args.mode == "daily":
        # Start daily scheduler
        scheduler.start_daily_scheduler()
        
    elif args.mode == "daemon":
        # Start as background daemon
        scheduler.start_background_daemon()
        
    elif args.mode == "test":
        # Run test cycle
        success = scheduler.run_test_cycle()
        sys.exit(0 if success else 1)
        
    elif args.mode == "status":
        # Show status
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
