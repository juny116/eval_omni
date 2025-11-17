from typing import Any
from openai import OpenAI, AsyncOpenAI, BadRequestError
from transformers import AutoTokenizer


class APIModel:
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str = None,
        max_retry: int = 5,
        timeout: int = 4000,
        base_url: str = "https://api.openai.com/v1",
        organization: str = None,
        project: str = None,
        chat_template: str = None,
    ):
        self.model = model
        self.api_key = api_key
        self.max_retry = max_retry
        self.timeout = timeout
        self.base_url = base_url
        # only if it is not openai api
        if self.base_url == "https://api.openai.com/v1":
            self.openai_api = True
        else:
            self.openai_api = False

        if not self.openai_api:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            if chat_template is not None:
                print(f"[INFO] Using custom chat template from {chat_template}")
                with open(chat_template, "r") as f:
                    self.tokenizer.chat_template = f.read()

        self.client = OpenAI(
            api_key=api_key,
            max_retries=self.max_retry,
            timeout=self.timeout,
            base_url=self.base_url,
            organization=organization,
            project=project,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            max_retries=self.max_retry,
            timeout=self.timeout,
            base_url=self.base_url,
            organization=organization,
            project=project,
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.generate(*args, **kwds)

    def generate(
        self,
        messages: list,
        max_tokens: int = 100,
        temperature: float = 1,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list = None,
        include_stop_str_in_output: bool = False,
        n: int = 1,
        **client_kwargs,
    ):
        if not self.openai_api:
            input_token_count = len(
                self.tokenizer.encode(
                    self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                )
            )
            print("orig messages:", messages)
            print("DEBUG: Applying chat template")
            print(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))  # DEBUG
            print("Done DEBUG")
            max_tokens = max_tokens - input_token_count
            if max_tokens < 0:
                max_tokens = 0

        messages = self._convert_message(messages)
        # Build parameters dict, sending only non‑None values
        params = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {
                "top_k": top_k,
                "min_p": min_p,
                "include_stop_str_in_output": include_stop_str_in_output,
            },
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            **client_kwargs,
        }
        if stop is not None:
            params["stop"] = stop
       
        # catch error and retry
        for i in range(self.max_retry):
            try:
                response = self.client.chat.completions.create(**params)
                return self._return_format(messages, response)
            except Exception as e:
                print(
                    f"[ERROR] OpenAI API call failed on attempt {i + 1}/{self.max_retry}: {e.__class__.__name__} - {e}"
                )
                if i == self.max_retry - 1:
                    return {"error": str(e)}

    async def agenerate(
        self,
        messages: list,
        max_tokens: int = 100,
        max_new_tokens: int = None,
        temperature: float = 1,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        include_stop_str_in_output: bool = False,
        stop_token_ids: list = None,
        stop: list = None,
        n: int = 1,
        logit_bias: dict = None,
        logprobs: int = None,
        **client_kwargs,
    ):
        if not self.openai_api:
            input_token_count = len(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
            if max_new_tokens is not None:
                max_tokens = max_new_tokens
            else:
                max_tokens = max_tokens - input_token_count
                if max_tokens < 0:
                    max_tokens = 0

        messages = self._convert_message(messages)
        params = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {
                "top_k": top_k,
                "min_p": min_p,
                "include_stop_str_in_output": include_stop_str_in_output,
                "stop_token_ids": stop_token_ids,
                "logprobs": logprobs,  # OpenAI does not support logprobs in chat completions
            },
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            "logit_bias": logit_bias,
            **client_kwargs,
        }

        if stop is not None:
            params["stop"] = stop
        response = await self.async_client.chat.completions.create(**params)
        return self._return_format(messages, response)

    def _convert_message(self, message: str) -> str:
        return message

    def _return_format(self, messages: list, response: dict) -> dict:
        # only consider case where n is 1
        # Combine reasoning content (if provided) with main content.
        assistant_content = response.choices[0].message.content

        reasoning_content = getattr(
            response.choices[0].message, "reasoning_content", None
        )

        if assistant_content is None and reasoning_content is None:
            print(f"[ERROR] OpenAI API call failed: {response.choices[0].message}")
        if assistant_content is None:
            assistant_content = ""

        if reasoning_content:
            if self.model == "LGAI-EXAONE/EXAONE-Deep-32B":
                assistant_content = (
                    f"<thought>\n{reasoning_content}</thought>\n" + assistant_content
                )
            else:
                assistant_content = (
                    f"<think>\n{reasoning_content}</think>\n" + assistant_content
                )
        # if the last message is not assistant append the assistant content
        if not messages or messages[-1]["role"] == "user":
            messages.append({"role": "assistant", "content": assistant_content})
        # if the last message is user append the assistant content
        elif messages[-1]["role"] == "assistant":
            messages[-1]["content"] += assistant_content

        response = {
            "messages": messages,
            "tokens": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            "logprobs": response.choices[
                0
            ].logprobs,  # OpenAI does not support logprobs in chat completions
            "error": None,
        }
        return response
