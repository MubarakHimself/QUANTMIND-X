"""Mock agent main module for testing"""
import sys
import time

def main():
    """Mock agent that keeps running"""
    print("Agent started successfully")
    sys.stdout.flush()
    
    # Keep running to allow tests to inspect container
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
