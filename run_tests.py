#!/usr/bin/env python3
"""
Test runner script that isolates from system ROS packages.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py tests/test_config.py  # Run specific test file
    python run_tests.py -m "not slow"      # Skip slow tests
    python run_tests.py -k "test_rotation" # Run tests matching pattern
"""

import subprocess
import sys
import os


def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up environment to isolate from system packages
    env = os.environ.copy()
    
    # Clear PYTHONPATH to avoid ROS conflicts
    env['PYTHONPATH'] = script_dir
    
    # Build pytest command
    pytest_args = [
        sys.executable, '-m', 'pytest',
        '-p', 'no:launch_testing',
        '-p', 'no:launch_pytest',
        '-p', 'no:launch_testing_ros',
        '-p', 'no:ament_pep257',
        '-p', 'no:ament_copyright',
        '-p', 'no:ament_lint',
        '-p', 'no:ament_xmllint',
        '-p', 'no:ament_flake8',
    ]
    
    # Add any additional arguments
    pytest_args.extend(sys.argv[1:])
    
    # Run pytest
    result = subprocess.run(pytest_args, env=env, cwd=script_dir)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()

