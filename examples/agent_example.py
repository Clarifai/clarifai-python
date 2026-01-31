"""Example demonstrating the Clarifai CLI Agent capabilities.

This shows how the agent system can execute CLI commands programmatically.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clarifai.cli.agent import ClarifaiAgent, parse_tool_calls_from_response

# Example 1: Initialize an agent
print("=" * 60)
print("EXAMPLE 1: Initializing the Agent")
print("=" * 60)

agent = ClarifaiAgent(pat="your_pat_here", user_id="your_user_id")

print("\nRegistered Tools:")
for tool in agent.tools.values():
    print(f"  - {tool.name}: {tool.description}")

# Example 2: List available tools
print("\n" + "=" * 60)
print("EXAMPLE 2: Tool Definitions for LLM")
print("=" * 60)

import json

tools_for_llm = agent.get_tools_for_llm()
print("\nTools formatted for LLM function calling:")
print(json.dumps(tools_for_llm[0], indent=2))

# Example 3: Parse tool calls from LLM response
print("\n" + "=" * 60)
print("EXAMPLE 3: Parsing Tool Calls from LLM Response")
print("=" * 60)

# Simulated LLM response with tool calls
llm_response = """I'll help you create a new app and list the models.

First, let me create the app:
<tool_call>{"tool": "create_app", "params": {"app_id": "my_vision_app", "name": "My Vision App"}}</tool_call>

Then list the models in that app:
<tool_call>{"tool": "list_models", "params": {"app_id": "my_vision_app"}}</tool_call>

This will set up everything you need!"""

tool_calls = parse_tool_calls_from_response(llm_response)
print(f"\nExtracted {len(tool_calls)} tool calls from response:")
for i, call in enumerate(tool_calls, 1):
    print(f"  {i}. Tool: {call['tool']}")
    print(f"     Params: {call['params']}")

# Example 4: Tool execution (would fail with dummy credentials, but shows the structure)
print("\n" + "=" * 60)
print("EXAMPLE 4: Tool Execution (with dummy credentials)")
print("=" * 60)

print("\nAttempting to list apps with dummy credentials:")
result = agent.execute_tool("list_apps", {})
print(f"Result: {json.dumps(result, indent=2)}")

# Example 5: Agent system prompt
print("\n" + "=" * 60)
print("EXAMPLE 5: Agent System Prompt for LLM")
print("=" * 60)

from clarifai.cli.agent import build_agent_system_prompt

system_prompt = build_agent_system_prompt(agent)
print("\nSystem prompt (first 500 chars):")
print(system_prompt[:500] + "...")

# Example 6: Chat integration concept
print("\n" + "=" * 60)
print("EXAMPLE 6: Integration with Chat")
print("=" * 60)

print("""
The agent integrates with the chat system as follows:

1. User sends a request to the chat interface
   $ clarifai chat
   You: Create an app called "test_app"

2. The chat system sends the request to the model with:
   - System prompt from build_agent_system_prompt(agent)
   - Available tools via get_tools_for_llm()
   - Conversation history

3. The model responds with natural language + tool calls:
   "I'll create that app for you.
    <tool_call>{"tool": "create_app", "params": {"app_id": "test_app"}}</tool_call>
    The app has been created successfully!"

4. The chat system:
   - Parses tool calls via parse_tool_calls_from_response()
   - Executes them via agent.execute_tool()
   - Gets a follow-up response from the model summarizing results
   - Shows the result to the user

This allows the agent to actually perform CLI actions!
""")

print("=" * 60)
print("See clarifai/cli/chat.py and clarifai/cli/agent.py for implementation")
print("=" * 60)
