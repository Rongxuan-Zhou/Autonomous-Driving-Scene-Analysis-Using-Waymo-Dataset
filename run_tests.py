#!/usr/bin/env python3
"""
Run all tests for the Autonomous Driving Scene Analysis project
"""

import unittest
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests in the tests directory"""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Add the project root to the Python path
    sys.path.append(str(script_dir))
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    print("Running tests for Autonomous Driving Scene Analysis Using Waymo Dataset")
    print("=" * 80)
    
    # Run the tests
    exit_code = run_tests()
    
    # Exit with the number of failures and errors
    sys.exit(exit_code)
