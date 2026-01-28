#!/usr/bin/python3
"""
Analyst Agent CLI - Main entry point
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import typer
from commands import app

if __name__ == "__main__":
    app()