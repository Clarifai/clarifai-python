"""Dummy OpenAI model implementation for testing."""

from typing import Any, Dict, Iterator, List

from clarifai.runners.models.openai_class import OpenAIModelClass


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    class ChatCompletions:
        def create(self, **kwargs):
            """Mock create method that returns a simple completion or stream."""
            if kwargs.get("stream", False):
                return MockCompletionStream(kwargs.get("messages", []))
            else:
                return MockCompletion(kwargs.get("messages", []))
    
    def __init__(self):
        self.chat = self.ChatCompletions()


class MockCompletion:
    """Mock completion object that mimics the OpenAI completion response structure."""
    
    class Choice:
        class Message:
            def __init__(self, content):
                self.content = content
                self.role = "assistant"
        
        def __init__(self, content):
            self.message = self.Message(content)
            self.finish_reason = "stop"
            self.index = 0
    
    def __init__(self, messages):
        # Generate a simple response based on the last message
        last_message = messages[-1] if messages else {"content": ""}
        response_text = f"Echo: {last_message.get('content', '')}"
        
        self.choices = [self.Choice(response_text)]
        self.usage = {
            "prompt_tokens": len(str(messages)),
            "completion_tokens": len(response_text),
            "total_tokens": len(str(messages)) + len(response_text)
        }
        self.id = "dummy-completion-id"
        self.created = 1234567890
        self.model = "dummy-model"


class MockCompletionStream:
    """Mock completion stream that mimics the OpenAI streaming response structure."""
    
    class Chunk:
        class Choice:
            class Delta:
                def __init__(self, content=None):
                    self.content = content
                    self.role = "assistant" if content is None else None
            
            def __init__(self, content=None):
                self.delta = self.Delta(content)
                self.finish_reason = None if content else "stop"
                self.index = 0
        
        def __init__(self, content=None):
            self.choices = [self.Choice(content)]
            self.id = "dummy-chunk-id"
            self.created = 1234567890
            self.model = "dummy-model"
    
    def __init__(self, messages):
        # Generate a simple response based on the last message
        last_message = messages[-1] if messages else {"content": ""}
        self.response_text = f"Echo: {last_message.get('content', '')}"
        # Divide the response into chunks of 5 characters
        self.chunks = [self.response_text[i:i+5] for i in range(0, len(self.response_text), 5)]
        self.current_chunk = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_chunk < len(self.chunks):
            chunk = self.Chunk(self.chunks[self.current_chunk])
            self.current_chunk += 1
            return chunk
        elif self.current_chunk == len(self.chunks):
            # Final chunk with empty content to indicate completion
            self.current_chunk += 1
            return self.Chunk()
        else:
            raise StopIteration


class DummyOpenAIModel(OpenAIModelClass):
    """Dummy OpenAI model implementation for testing."""
    
    def get_openai_client(self):
        """Return a mock OpenAI client."""
        return MockOpenAIClient()
    
    def _process_request(self, model, messages, temperature=1.0, max_tokens=None):
        """Override to handle the MockCompletion response format."""
        completion_args = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens is not None:
            completion_args["max_tokens"] = max_tokens
        
        # This returns our mock completion object
        return self.client.chat.completions.create(**completion_args).choices[0].message.content
    
    def _process_streaming_request(self, model, messages, temperature=1.0, max_tokens=None):
        """Override to handle the MockCompletionStream response format."""
        completion_args = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        if max_tokens is not None:
            completion_args["max_tokens"] = max_tokens
        
        # This returns our mock stream
        for chunk in self.client.chat.completions.create(**completion_args):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    # Additional example method that could be added for specific model implementations
    @OpenAIModelClass.method
    def test_method(self, prompt: str) -> str:
        """Test method that simply echoes the input."""
        return f"Test: {prompt}"