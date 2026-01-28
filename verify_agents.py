"""
Verification script for the Agent Framework.
"""
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

import asyncio

# Set dummy keys for structural verification
os.environ["OPENAI_API_KEY"] = "sk-dummy"
os.environ["ANTHROPIC_API_KEY"] = "sk-dummy"

from src.agents.implementations.analyst import create_analyst_agent
from src.agents.implementations.quant_code import create_quant_code_agent

async def verify():
    print("--- Verifying Agent Framework ---")
    
    # 1. Test Analyst
    print("Instantiating Analyst...")
    analyst = create_analyst_agent()
    print(f"Analyst Tools: Not applicable (Custom Graph)")
    # Structural check on the graph
    assert analyst.graph is not None
    print("Analyst Graph compiled successfully.")
    
    # 2. Test QuantCode
    print("Instantiating QuantCode...")
    quant_code = create_quant_code_agent()
    print(f"QuantCode Tools: {[t.name for t in quant_code.tools]}")
    assert "read_file" in [t.name for t in quant_code.tools]
    assert "update_todo_list" in [t.name for t in quant_code.tools]
    
    print("\n[SUCCESS] Custom graphs and base agents verified.")

if __name__ == "__main__":
    asyncio.run(verify())
