#!/usr/bin/env python3
"""
Simple test runner for PyVRP endpoint tests

This script runs the comprehensive PyVRP tests and provides quick setup.
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_server_availability(url: str, max_retries: int = 5) -> bool:
    """Check if the server is available"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        
        if i < max_retries - 1:
            print(f"Server not available, retrying in 2 seconds... ({i+1}/{max_retries})")
            time.sleep(2)
    
    return False

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False

def main():
    """Main runner"""
    print("PyVRP Endpoint Test Runner")
    print("=" * 40)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    test_script = current_dir / "test_pyvrp_comprehensive.py"
    
    if not test_script.exists():
        print(f"Error: test_pyvrp_comprehensive.py not found in {current_dir}")
        print("Please run this script from the directory containing the test script.")
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Configuration
    server_url = "http://localhost:8000"
    
    print(f"\nChecking server availability at {server_url}...")
    
    if not check_server_availability(server_url):
        print(f"""
❌ Server not available at {server_url}

To start the server, run:
1. cd backend
2. python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Or adjust the BASE_URL in test_pyvrp_comprehensive.py if using a different URL.
""")
        return 1
    
    print("✅ Server is available!")
    
    # Run the comprehensive test
    print("\n" + "="*50)
    print("RUNNING COMPREHENSIVE PYVRP TESTS")
    print("="*50)
    
    try:
        result = subprocess.run([sys.executable, "test_pyvrp_comprehensive.py"], 
                               capture_output=True, text=True)
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Tests completed successfully!")
        else:
            print(f"\n❌ Tests failed with return code {result.returncode}")
        
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())