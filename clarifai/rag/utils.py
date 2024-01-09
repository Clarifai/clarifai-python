from typing import List


## TODO: Make this token-aware.
def convert_messages_to_str(messages: List[dict]) -> str:
  """convert messages in OpenAI API format into a single string.

    Args:
        messages List[dict]: A list of dictionary in the following format:
        ```
        [
          {"role": "user", "content": "Hello there."},
          {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
          {"role": "user", "content": "Can you explain LLMs in plain English?"},
        ]
        ```
    """
  final_str = ""
  for msg in messages:
    if "role" in msg and "content" in msg:
      role = msg.get("role", "")
      content = msg.get("content", "")
      final_str += f"\n\n{role}: {content}"
  return final_str


def format_assistant_message(raw_text: str) -> dict:
  return {"role": "assistant", "content": raw_text}
