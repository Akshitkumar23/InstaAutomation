@echo off
cd /d "c:\Users\akshi\Desktop\instaAutomate\InstaAutomation"
call .venv\Scripts\activate
python scheduler.py --mode once >> scheduler.log 2>&1
