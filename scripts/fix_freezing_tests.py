#!/usr/bin/env python3
"""
Script to identify and fix freezing tests in the QuantMindX test suite.

This script:
1. Identifies tests that might freeze (long-running async operations)
2. Adds proper timeouts to prevent freezing
3. Fixes common async test issues
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_test_files() -> List[Path]:
    """Find all test files in the tests directory."""
    test_dir = Path("tests")
    return list(test_dir.rglob("test_*.py"))


def check_for_freezing_patterns(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Check a test file for patterns that might cause freezing.
    
    Returns list of (line_number, pattern, issue_description)
    """
    issues = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        # Check for infinite loops
        if re.search(r'while\s+True:', line):
            issues.append((i, line.strip(), "Infinite loop without timeout"))
        
        # Check for long sleep operations
        sleep_match = re.search(r'(asyncio\.sleep|time\.sleep)\((\d+)\)', line)
        if sleep_match:
            sleep_time = int(sleep_match.group(2))
            if sleep_time > 10:
                issues.append((i, line.strip(), f"Long sleep operation ({sleep_time}s)"))
        
        # Check for async tests without timeout marker
        if re.search(r'async\s+def\s+test_', line):
            # Check if previous lines have @pytest.mark.timeout
            has_timeout = False
            for j in range(max(0, i-5), i):
                if '@pytest.mark.timeout' in lines[j]:
                    has_timeout = True
                    break
            
            if not has_timeout:
                issues.append((i, line.strip(), "Async test without explicit timeout"))
        
        # Check for blocking operations in async context
        if 'async def' in ''.join(lines[max(0, i-10):i]):
            if re.search(r'\.get\(\)|\.result\(\)|\.join\(\)', line):
                issues.append((i, line.strip(), "Blocking operation in async context"))
    
    return issues


def add_timeout_to_test(file_path: Path, test_line: int, timeout: int = 30):
    """Add timeout decorator to a test function."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the test function definition
    for i in range(test_line - 1, max(0, test_line - 10), -1):
        if lines[i].strip().startswith('def test_') or lines[i].strip().startswith('async def test_'):
            # Check if timeout already exists
            has_timeout = False
            for j in range(max(0, i-5), i):
                if '@pytest.mark.timeout' in lines[j]:
                    has_timeout = True
                    break
            
            if not has_timeout:
                # Add timeout decorator
                indent = len(lines[i]) - len(lines[i].lstrip())
                timeout_line = ' ' * indent + f'@pytest.mark.timeout({timeout})\n'
                lines.insert(i, timeout_line)
                
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                
                return True
    
    return False


def fix_long_sleeps(file_path: Path):
    """Replace long sleep operations with shorter ones or remove them."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace long sleeps with shorter ones
    modified = False
    
    # Replace sleep(60) or higher with sleep(1)
    new_content = re.sub(
        r'(asyncio\.sleep|time\.sleep)\(([6-9]\d|\d{3,})\)',
        r'\1(1)',
        content
    )
    
    if new_content != content:
        modified = True
        with open(file_path, 'w') as f:
            f.write(new_content)
    
    return modified


def add_asyncio_timeout_wrapper(file_path: Path):
    """Add asyncio.wait_for wrapper to long-running operations."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    
    for i, line in enumerate(lines):
        # Look for await statements without timeout
        if 'await ' in line and 'asyncio.wait_for' not in line:
            # Check if it's a potentially long operation
            if any(keyword in line for keyword in ['stream', 'connect', 'start', 'run']):
                # Add timeout wrapper
                indent = len(line) - len(line.lstrip())
                await_match = re.search(r'await\s+(.+)', line)
                if await_match:
                    operation = await_match.group(1).strip()
                    new_line = ' ' * indent + f'await asyncio.wait_for({operation}, timeout=30)\n'
                    lines[i] = new_line
                    modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
    
    return modified


def main():
    """Main function to fix freezing tests."""
    print("=" * 70)
    print("QuantMindX Test Freezing Fix Script")
    print("=" * 70)
    
    test_files = find_test_files()
    print(f"\nFound {len(test_files)} test files")
    
    total_issues = 0
    fixed_files = []
    
    for test_file in test_files:
        print(f"\nAnalyzing: {test_file}")
        
        issues = check_for_freezing_patterns(test_file)
        
        if issues:
            print(f"  Found {len(issues)} potential issues:")
            for line_num, pattern, description in issues:
                print(f"    Line {line_num}: {description}")
                print(f"      {pattern}")
                total_issues += 1
            
            # Apply fixes
            print(f"\n  Applying fixes...")
            
            # Fix long sleeps
            if fix_long_sleeps(test_file):
                print(f"    ✓ Fixed long sleep operations")
                fixed_files.append(test_file)
            
            # Add timeouts to async tests
            for line_num, pattern, description in issues:
                if "Async test without explicit timeout" in description:
                    if add_timeout_to_test(test_file, line_num):
                        print(f"    ✓ Added timeout to test at line {line_num}")
                        if test_file not in fixed_files:
                            fixed_files.append(test_file)
        else:
            print(f"  ✓ No issues found")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total issues found: {total_issues}")
    print(f"Files modified: {len(fixed_files)}")
    
    if fixed_files:
        print("\nModified files:")
        for file in fixed_files:
            print(f"  - {file}")
    
    print("\n" + "=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print("1. Install pytest-timeout: pip install pytest-timeout")
    print("2. Run tests with timeout: pytest --timeout=30")
    print("3. For slow tests, use: @pytest.mark.timeout(120)")
    print("4. Review modified files and test manually")
    print("5. Consider using asyncio.wait_for() for long operations")
    
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
