import os
from threading import Thread
from typing import Iterator, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import openai_response
from clarifai.utils.logging import logger


class HuggingFaceModel(OpenAIModelClass):
    client = True
    model = True

    def load_model(self):
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info(f"Using device: {self.device}")

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        config = builder.config
        stage = config["checkpoints"]["when"]
        checkpoints = config["checkpoints"]["repo_id"]
        if stage in ["build", "runtime"]:
            checkpoints = builder.download_checkpoints(stage=stage)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoints)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoints,
            low_cpu_mem_usage=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        self.streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (higher = more random).",
        ),
        top_p: float = Param(
            default=0.8,
            description="Nucleus sampling threshold.",
        ),
    ) -> str:
        """Return a single completion."""
        messages = chat_history if chat_history else []
        if prompt:
            messages.append({"role": "user", "content": prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_tokens = output[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @OpenAIModelClass.method
    def generate(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (higher = more random).",
        ),
        top_p: float = Param(
            default=0.8,
            description="Nucleus sampling threshold.",
        ),
    ) -> Iterator[str]:
        """Stream a completion response."""
        messages = chat_history if chat_history else []
        if prompt:
            messages.append({"role": "user", "content": prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = {
            "input_ids": inputs,
            "do_sample": True,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": self.streamer,
        }
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for chunk in openai_response(self.streamer):
            if (
                'choices' in chunk
                and 'delta' in chunk['choices'][0]
                and 'content' in chunk['choices'][0]['delta']
            ):
                yield chunk['choices'][0]['delta']['content']
        thread.join()
