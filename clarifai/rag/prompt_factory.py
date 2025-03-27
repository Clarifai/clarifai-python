from enum import Enum
from typing import Any, Callable, List, Optional

import requests


def default_pt(messages):
  return " ".join(message["content"] for message in messages)


# alpaca prompt template - for models like mythomax, etc.
def alpaca_pt(messages):
  prompt = custom_prompt(
      role_dict={
          "system": {
              "pre_message": "### Instruction:\n",
              "post_message": "\n\n",
          },
          "user": {
              "pre_message": "### Instruction:\n",
              "post_message": "\n\n",
          },
          "assistant": {
              "pre_message": "### Response:\n",
              "post_message": "\n\n"
          },
      },
      bos_token="<s>",
      eos_token="</s>",
      messages=messages,
  )
  return prompt


# Llama2 prompt template
def llama_2_chat_pt(messages):
  prompt = custom_prompt(
      role_dict={
          "system": {
              "pre_message": "[INST] <<SYS>>\n",
              "post_message": "\n<</SYS>>\n [/INST]\n",
          },
          "user": {  # follow this format https://github.com/facebookresearch/llama/blob/77062717054710e352a99add63d160274ce670c6/llama/generation.py#L348
              "pre_message": "[INST] ",
              "post_message": " [/INST]\n",
          },
          "assistant": {
              "post_message": "\n"  # follows this - https://replicate.com/blog/how-to-prompt-llama
          },
      },
      messages=messages,
      bos_token="<s>",
      eos_token="</s>",
  )
  return prompt


def ollama_pt(
    model, messages
):  # https://github.com/jmorganca/ollama/blob/af4cf55884ac54b9e637cd71dadfe9b7a5685877/docs/modelfile.md#template
  if "instruct" in model:
    prompt = custom_prompt(
        role_dict={
            "system": {
                "pre_message": "### System:\n",
                "post_message": "\n"
            },
            "user": {
                "pre_message": "### User:\n",
                "post_message": "\n",
            },
            "assistant": {
                "pre_message": "### Response:\n",
                "post_message": "\n",
            },
        },
        final_prompt_value="### Response:",
        messages=messages,
    )
  elif "llava" in model:
    prompt = ""
    images = []
    for message in messages:
      if isinstance(message["content"], str):
        prompt += message["content"]
      elif isinstance(message["content"], list):
        for element in message["content"]:
          if isinstance(element, dict):
            if element["type"] == "text":
              prompt += element["text"]
            elif element["type"] == "image_url":
              image_url = element["image_url"]["url"]
              images.append(image_url)
    return {"prompt": prompt, "images": images}
  else:
    prompt = "".join(m["content"] if isinstance(m["content"], str) is str else "".join(
        m["content"]) for m in messages)
  return prompt


def mistral_instruct_pt(messages):
  # Following the Mistral example's https://huggingface.co/docs/transformers/main/chat_templating
  prompt = custom_prompt(
      initial_prompt_value="<s>",
      role_dict={
          "system": {
              "pre_message": "[INST] \n",
              "post_message": " [/INST]\n",
          },
          "user": {
              "pre_message": "[INST] ",
              "post_message": " [/INST]\n"
          },
          "assistant": {
              "pre_message": " ",
              "post_message": " "
          },
      },
      final_prompt_value="</s>",
      messages=messages,
  )
  return prompt


# Falcon prompt template - from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L110
def falcon_instruct_pt(messages):
  prompt = ""
  for message in messages:
    if message["role"] == "system":
      prompt += message["content"]
    else:
      prompt += (
          message["role"] + ":" + message["content"].replace("\r\n", "\n").replace("\n\n", "\n"))
      prompt += "\n\n"

  return prompt


def falcon_chat_pt(messages):
  prompt = ""
  for message in messages:
    if message["role"] == "system":
      prompt += "System: " + message["content"]
    elif message["role"] == "assistant":
      prompt += "Falcon: " + message["content"]
    elif message["role"] == "user":
      prompt += "User: " + message["content"]

  return prompt


# MPT prompt template - from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L110
def mpt_chat_pt(messages):
  prompt = ""
  for message in messages:
    if message["role"] == "system":
      prompt += "<|im_start|>system" + message["content"] + "<|im_end|>" + "\n"
    elif message["role"] == "assistant":
      prompt += "<|im_start|>assistant" + message["content"] + "<|im_end|>" + "\n"
    elif message["role"] == "user":
      prompt += "<|im_start|>user" + message["content"] + "<|im_end|>" + "\n"
  return prompt


# WizardCoder prompt template - https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0#prompt-format
def wizardcoder_pt(messages):
  prompt = ""
  for message in messages:
    if message["role"] == "system":
      prompt += message["content"] + "\n\n"
    elif message["role"] == "user":  # map to 'Instruction'
      prompt += "### Instruction:\n" + message["content"] + "\n\n"
    elif message["role"] == "assistant":  # map to 'Response'
      prompt += "### Response:\n" + message["content"] + "\n\n"
  return prompt


# Phind-CodeLlama prompt template - https://huggingface.co/Phind/Phind-CodeLlama-34B-v2#how-to-prompt-the-model
def phind_codellama_pt(messages):
  prompt = ""
  for message in messages:
    if message["role"] == "system":
      prompt += "### System Prompt\n" + message["content"] + "\n\n"
    elif message["role"] == "user":
      prompt += "### User Message\n" + message["content"] + "\n\n"
    elif message["role"] == "assistant":
      prompt += "### Assistant\n" + message["content"] + "\n\n"
  return prompt


# Anthropic template
def claude_2_1_pt(
    messages: list,):  # format - https://docs.anthropic.com/claude/docs/how-to-use-system-prompts
  """
    Claude v2.1 allows system prompts (no Human: needed), but requires it be followed by Human:
    - you can't just pass a system message
    - you can't pass a system message and follow that with an assistant message
    if system message is passed in, you can only do system, human, assistant or system, human

    if a system message is passed in and followed by an assistant message, insert a blank human message between them.

    Additionally, you can "put words in Claude's mouth" by ending with an assistant message.
    See: https://docs.anthropic.com/claude/docs/put-words-in-claudes-mouth
    """

  class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "

  prompt = ""
  for idx, message in enumerate(messages):
    if message["role"] == "user":
      prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
    elif message["role"] == "system":
      prompt += f"{message['content']}"
    elif message["role"] == "assistant":
      if idx > 0 and messages[idx - 1]["role"] == "system":
        prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}"  # Insert a blank human message
      prompt += f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
  if messages[-1]["role"] != "assistant":
    prompt += f"{AnthropicConstants.AI_PROMPT.value}"  # prompt must end with \"\n\nAssistant: " turn
  return prompt


### TOGETHER AI


def get_model_info(token, model):
  try:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://api.together.xyz/models/info", headers=headers)
    if response.status_code == 200:
      model_info = response.json()
      for m in model_info:
        if m["name"].lower().strip() == model.strip():
          return m["config"].get("prompt_format", None), m["config"].get("chat_template", None)
      return None, None
    else:
      return None, None
  except Exception:  # safely fail a prompt template request
    return None, None


def format_prompt_togetherai(messages, prompt_format, chat_template):
  if prompt_format is None:
    return default_pt(messages)

  human_prompt, assistant_prompt = prompt_format.split("{prompt}")

  if prompt_format is not None:
    prompt = custom_prompt(
        role_dict={},
        messages=messages,
        initial_prompt_value=human_prompt,
        final_prompt_value=assistant_prompt,
    )
  else:
    prompt = default_pt(messages)
  return prompt


###


def anthropic_pt(
    messages: list,):  # format - https://docs.anthropic.com/claude/reference/complete_post
  """
    You can "put words in Claude's mouth" by ending with an assistant message.
    See: https://docs.anthropic.com/claude/docs/put-words-in-claudes-mouth
    """

  class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "

  prompt = ""
  for idx, message in enumerate(
      messages):  # needs to start with `\n\nHuman: ` and end with `\n\nAssistant: `
    if message["role"] == "user":
      prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
    elif message["role"] == "system":
      prompt += f"{AnthropicConstants.HUMAN_PROMPT.value}<admin>{message['content']}</admin>"
    else:
      prompt += f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
    if (idx == 0 and
        message["role"] == "assistant"):  # ensure the prompt always starts with `\n\nHuman: `
      prompt = f"{AnthropicConstants.HUMAN_PROMPT.value}" + prompt
  if messages[-1]["role"] != "assistant":
    prompt += f"{AnthropicConstants.AI_PROMPT.value}"
  return prompt


def _load_image_from_url(image_url):
  # try:
  # except:
  #     raise Exception(
  #         "gemini image conversion failed please run `pip install Pillow`"
  #     )
  from io import BytesIO

  from PIL import Image

  try:
    # Send a GET request to the image URL
    response = requests.get(image_url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Check the response's content type to ensure it is an image
    content_type = response.headers.get("content-type")
    if not content_type or "image" not in content_type:
      raise ValueError(f"URL does not point to a valid image (content-type: {content_type})")

    # Load the image from the response content
    return Image.open(BytesIO(response.content))

  except requests.RequestException as e:
    raise Exception(f"Request failed: {e}")
  except Exception as e:
    raise e


def _gemini_vision_convert_messages(messages: list):
  """
    Converts given messages for GPT-4 Vision to Gemini format.

    Args:
        messages (list): The messages to convert. Each message can be a dictionary with a "content" key. The content can be a string or a list of elements. If it is a string, it will be concatenated to the prompt. If it is a list, each element will be processed based on its type:
            - If the element is a dictionary with a "type" key equal to "text", its "text" value will be concatenated to the prompt.
            - If the element is a dictionary with a "type" key equal to "image_url", its "image_url" value will be added to the list of images.

    Returns:
        tuple: A tuple containing the prompt (a string) and the processed images (a list of objects representing the images).
    """
  # try:
  from PIL import Image

  # except:
  #     raise Exception(
  #         "gemini image conversion failed please run `pip install Pillow`"
  #     )

  try:
    # given messages for gpt-4 vision, convert them for gemini
    # https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb
    prompt = ""
    images = []
    for message in messages:
      if isinstance(message["content"], str):
        prompt += message["content"]
      elif isinstance(message["content"], list):
        for element in message["content"]:
          if isinstance(element, dict):
            if element["type"] == "text":
              prompt += element["text"]
            elif element["type"] == "image_url":
              image_url = element["image_url"]["url"]
              images.append(image_url)
    # processing images passed to gemini
    processed_images = []
    for img in images:
      if "https:/" in img:
        # Case 1: Image from URL
        image = _load_image_from_url(img)
        processed_images.append(image)
      else:
        # Case 2: Image filepath (e.g. temp.jpeg) given
        image = Image.open(img)
        processed_images.append(image)
    content = [prompt] + processed_images
    return content
  except Exception as e:
    raise e


def gemini_text_image_pt(messages: list):
  """
    {
        "contents":[
            {
            "parts":[
                {"text": "What is this picture?"},
                {
                "inline_data": {
                    "mime_type":"image/jpeg",
                    "data": "'$(base64 -w0 image.jpg)'"
                }
                }
            ]
            }
        ]
    }
    """
  # try:
  #     pass
  # except:
  #     raise Exception(
  #         "Importing google.generativeai failed, please run 'pip install -q google-generativeai"
  #     )

  prompt = ""
  images = []
  for message in messages:
    if isinstance(message["content"], str):
      prompt += message["content"]
    elif isinstance(message["content"], list):
      for element in message["content"]:
        if isinstance(element, dict):
          if element["type"] == "text":
            prompt += element["text"]
          elif element["type"] == "image_url":
            image_url = element["image_url"]["url"]
            images.append(image_url)

  content = [prompt] + images
  return content


# Function call template
def function_call_prompt(messages: list, functions: list):
  function_prompt = ("Produce JSON OUTPUT ONLY! The following functions are available to you:")
  for function in functions:
    function_prompt += f"""\n{function}\n"""

  function_added_to_prompt = False
  for message in messages:
    if "system" in message["role"]:
      message["content"] += f"""{function_prompt}"""
      function_added_to_prompt = True

  if function_added_to_prompt is False:
    messages.append({"role": "system", "content": f"""{function_prompt}"""})

  return messages


# Custom prompt template
def custom_prompt(
    role_dict: dict,
    messages: list,
    initial_prompt_value: str = "",
    final_prompt_value: str = "",
    bos_token: str = "",
    eos_token: str = "",
):
  prompt = bos_token + initial_prompt_value
  bos_open = True
  ## a bos token is at the start of a system / human message
  ## an eos token is at the end of the assistant response to the message
  for message in messages:
    role = message["role"]

    if role in ["system", "human"] and not bos_open:
      prompt += bos_token
      bos_open = True

    pre_message_str = (role_dict[role]["pre_message"]
                       if role in role_dict and "pre_message" in role_dict[role] else "")
    post_message_str = (role_dict[role]["post_message"]
                        if role in role_dict and "post_message" in role_dict[role] else "")
    prompt += pre_message_str + message["content"] + post_message_str

    if role == "assistant":
      prompt += eos_token
      bos_open = False

  prompt += final_prompt_value
  return prompt


def prompt_factory(
    model: str,
    messages: List[Any],
    custom_llm_provider: Optional[str] = None,
    api_key: Optional[str] = None,
):
  """Decorator to apply prompt formatting based on model and provider."""

  def decorator(func: Callable[[List[Any]], Any]) -> Callable[[List[Any]], Any]:

    def wrapper(messages: List[Any]) -> Any:
      nonlocal model
      model = model.lower()
      if custom_llm_provider == "ollama":
        return ollama_pt(model=model, messages=messages)
      elif custom_llm_provider == "anthropic":
        if any(_ in model for _ in ["claude-2.1", "claude-v2:1"]):
          return claude_2_1_pt(messages=messages)
        else:
          return anthropic_pt(messages=messages)
      elif custom_llm_provider == "together_ai":
        prompt_format, chat_template = get_model_info(token=api_key, model=model)
        return format_prompt_togetherai(
            messages=messages, prompt_format=prompt_format, chat_template=chat_template)
      elif custom_llm_provider == "gemini":
        if model == "gemini-pro-vision":
          return _gemini_vision_convert_messages(messages=messages)
        else:
          return gemini_text_image_pt(messages=messages)
      try:
        if "meta-llama/llama-2" in model and "chat" in model:
          return llama_2_chat_pt(messages=messages)
        elif "tiiuae/falcon" in model:
          if model == "tiiuae/falcon-180B-chat":
            return falcon_chat_pt(messages=messages)
          elif "instruct" in model:
            return falcon_instruct_pt(messages=messages)
        elif "mosaicml/mpt" in model:
          if "chat" in model:
            return mpt_chat_pt(messages=messages)
        elif "codellama/codellama" in model or "togethercomputer/codellama" in model:
          if "instruct" in model:
            return llama_2_chat_pt(messages=messages)
        elif "wizardlm/wizardcoder" in model:
          return wizardcoder_pt(messages=messages)
        elif "phind/phind-codellama" in model:
          return phind_codellama_pt(messages=messages)
        elif "togethercomputer/llama-2" in model and ("instruct" in model or "chat" in model):
          return llama_2_chat_pt(messages=messages)
        elif model in [
            "gryphe/mythomax-l2-13b",
            "gryphe/mythomix-l2-13b",
            "gryphe/mythologic-l2-13b",
        ]:
          return alpaca_pt(messages=messages)
        elif "mistral" in model:
          return mistral_instruct_pt(messages=messages)
      except Exception:
        return default_pt(messages=messages)

    return wrapper

  return decorator
