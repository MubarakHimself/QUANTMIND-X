#!/usr/bin/env python3
"""
Initialize Git repository structure for Assets Hub.

Creates the directory structure for templates and skills at:
    /data/git/assets-hub/templates/{category}/
    /data/git/assets-hub/skills/{category}/

Adds .gitignore for generated_bots/ temporary files.

Usage:
    python scripts/init_git_assets_hub.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def init_git_repository():
    """Initialize Git repository structure for Assets Hub."""
    from src.agents.integrations.git_client import GitClient

    git_repo_path = PROJECT_ROOT / "data" / "git" / "assets-hub"
    git_client = GitClient(git_repo_path)

    print("=" * 60)
    print("Initializing Git Repository for Assets Hub")
    print("=" * 60)

    # Initialize repository
    print("\n1. Initializing Git repository...")
    if not git_client.init_repo():
        print("   Error: Failed to initialize Git repository")
        return False

    # Configure user
    print("\n2. Configuring Git user...")
    if not git_client.configure_user():
        print("   Error: Failed to configure Git user")
        return False

    # Create directory structure
    print("\n3. Creating directory structure...")

    # Template categories
    template_categories = [
        "trend_following",
        "mean_reversion",
        "breakout",
        "momentum",
        "arbitrage"
    ]

    for category in template_categories:
        category_path = git_repo_path / "templates" / category
        category_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: templates/{category}/")

    # Skill categories
    skill_categories = [
        "trading_skills",
        "system_skills",
        "data_skills"
    ]

    for category in skill_categories:
        category_path = git_repo_path / "skills" / category
        category_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: skills/{category}/")

    # Create generated_bots directory with .gitignore
    print("\n4. Setting up generated_bots directory...")
    generated_bots_path = git_repo_path / "generated_bots"
    generated_bots_path.mkdir(parents=True, exist_ok=True)

    gitignore_path = generated_bots_path / ".gitignore"
    gitignore_content = """# Generated trading bots
# These are temporary outputs from code generation
*.mq5
*.py
*.json
*.log
*/

# Keep directory structure
!.gitignore
"""
    gitignore_path.write_text(gitignore_content)
    print(f"   Created: generated_bots/.gitignore")

    # Create README
    readme_path = git_repo_path / "README.md"
    readme_content = """# Assets Hub

This Git repository stores raw source files for the QuantMindX Assets Hub.

## Structure

- `templates/`: Algorithm template files organized by category
  - `trend_following/`: Trend-following strategy templates
  - `mean_reversion/`: Mean-reversion strategy templates
  - `breakout/`: Breakout strategy templates
  - `momentum/`: Momentum-based strategy templates
  - `arbitrage/`: Arbitrage strategy templates

- `skills/`: Agentic skill definitions organized by category
  - `trading_skills/`: Trading-related skills (indicators, signals, risk management)
  - `system_skills/`: System-level skills (file operations, API calls, logging)
  - `data_skills/`: Data processing skills (fetching, cleaning, resampling)

- `generated_bots/`: Temporary storage for generated trading bots
  - Contents are gitignored (see .gitignore)

- `configuration/`: Configuration files for backtesting and deployment

## Usage

Templates and skills are managed through the GitClient class:
- `read_template(category, name)`: Read a template file
- `write_template(category, name, content)`: Write a template file
- `commit_template(category, name, message)`: Commit template changes

## Commit Message Format

Template commits follow this format:
```
Strategy: {name}, Backtest: Sharpe={sharpe}, Drawdown={drawdown}%
```

Example:
```
Strategy: RSI Mean Reversion, Backtest: Sharpe=1.8, Drawdown=12%
```
"""
    readme_path.write_text(readme_content)
    print(f"   Created: README.md")

    # Create configuration directory
    config_path = git_repo_path / "configuration"
    config_path.mkdir(parents=True, exist_ok=True)
    print(f"   Created: configuration/")

    # Initial commit
    print("\n5. Creating initial commit...")
    import subprocess
    try:
        subprocess.run(["git", "add", "."], cwd=git_repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit: Initialize Assets Hub repository structure"],
            cwd=git_repo_path,
            capture_output=True,
            check=True
        )
        print("   Initial commit created successfully")
    except subprocess.CalledProcessError as e:
        # May fail if nothing to commit
        if "nothing to commit" not in str(e.stderr).lower():
            print(f"   Warning: Could not create initial commit: {e.stderr}")

    print("\n" + "=" * 60)
    print("Git repository structure initialized successfully")
    print(f"Location: {git_repo_path}")
    print("=" * 60)

    return True


def main():
    """Main function."""
    try:
        success = init_git_repository()
        if success:
            print("\nNext steps:")
            print("1. Add template files to templates/{category}/")
            print("2. Add skill definitions to skills/{category}/")
            print("3. Commit changes with descriptive messages")
            sys.exit(0)
        else:
            print("\nInitialization failed")
            sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
