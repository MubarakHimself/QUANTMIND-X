"""
Analyst Verification Script
"""
import asyncio
from src.agents.implementations.analyst import create_analyst_agent

async def main():
    print("--- Testing Analyst Synthesis Loop ---")
    analyst = create_analyst_agent()
    
    # We ask it to run its internal graph for the first file it finds in data/nprd/
    # Since the nodes are designed to find files, we just need to trigger the graph.
    print("\nTriggering Strategy Synthesis...")
    response = await analyst.ainvoke("Process the RSI Mean Reversion strategy from data/nprd/rsi_reversion.txt")
    
    print(f"\nAnalyst Response:\n{response}")
    
    import os
    trd_path = "docs/trds/generated_strategy.md"
    if os.path.exists(trd_path):
        print(f"\n[SUCCESS] TRD generated at: {trd_path}")
        with open(trd_path, "r") as f:
            print("\nTRD PREVIEW:")
            print("-" * 20)
            print(f.read()[:500] + "...")
    else:
        print("\n[FAILURE] TRD not found.")

if __name__ == "__main__":
    asyncio.run(main())
