#!/usr/bin/env python3
"""NBA Prediction Agent — fast-agent application.

Interactive AI agent for NBA analysis and predictions.
Uses trained XGBoost models via two MCP servers:
  - nba_data:    team/player/schedule/standings queries
  - nba_predict: game winner, point spread, player projections, MVP race

Usage (via fast-agent CLI):
    cd agent
    fast-agent go -i instruction.md -c config.yaml --servers nba_data,nba_predict

    # Single query:
    fast-agent go -i instruction.md -c config.yaml --servers nba_data,nba_predict \
        -m "Who wins tonight: Lakers vs Celtics?"

Usage (programmatic — requires fast-agent-mcp in your Python environment):
    python agent.py
"""

import asyncio

try:
    from mcp_agent.core.fastagent import FastAgent

    fast = FastAgent("NBA Prediction Agent")

    @fast.agent(
        name="nba_analyst",
        instruction="instruction.md",
        servers=["nba_data", "nba_predict"],
        model="sonnet",
        human_input=True,
    )
    async def main():
        async with fast.run() as agent:
            await agent()

    if __name__ == "__main__":
        asyncio.run(main())

except ImportError:
    if __name__ == "__main__":
        print("fast-agent-mcp not found in this Python environment.")
        print()
        print("Use the fast-agent CLI instead:")
        print("  cd agent")
        print("  fast-agent go -i instruction.md -c config.yaml "
              "--servers nba_data,nba_predict")
        print()
        print("Or for a single query:")
        print('  fast-agent go -i instruction.md -c config.yaml '
              '--servers nba_data,nba_predict '
              '-m "Who wins: Lakers vs Celtics?"')
