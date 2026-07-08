from typing import List, Union, Optional, Literal
import dataclasses
import os

from tenacity import (
    RetryCallState,
    retry,
    stop_after_delay,  # type: ignore
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai
from rich.console import Console
from rich.panel import Panel

MessageRole = Literal["system", "user", "assistant"]
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "120"))
DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "4"))
DEFAULT_RETRY_MAX_WAIT = float(os.getenv("OPENAI_RETRY_MAX_WAIT", "30"))
DEFAULT_RETRY_MAX_SECONDS = float(os.getenv("OPENAI_RETRY_MAX_SECONDS", "300"))


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


def _choice_value(choice, key: str, default=None):
    if isinstance(choice, dict):
        return choice.get(key, default)
    return getattr(choice, key, default)


def _message_content(message) -> str:
    if message is None:
        return ""
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", None) or ""


def _message_debug(message) -> str:
    if message is None:
        return "message=None"
    if hasattr(message, "to_dict_recursive"):
        return str(message.to_dict_recursive())
    if isinstance(message, dict):
        return str(message)
    return repr(message)


def _log_chat_choice(choice) -> None:
    finish_reason = _choice_value(choice, "finish_reason", "unknown")
    message = _choice_value(choice, "message")
    content = _message_content(message)
    lines = [
        f"finish_reason: {finish_reason}",
        f"content_length: {len(content)}",
    ]
    if not content.strip():
        lines.append(f"message: {_message_debug(message)}")
    panel = Panel("\n".join(lines), title="Chat response metadata", border_style="blue")
    try:
        from utils import print_v
        print_v(panel)
    except Exception:
        Console().print(panel)


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    lines = [
        f"attempt: {retry_state.attempt_number}",
        f"next_wait_seconds: {round(wait, 2)}",
    ]
    if exc is not None:
        lines.append(f"exception: {type(exc).__name__}: {exc}")
    panel = Panel("\n".join(lines), title="Model API retry", border_style="yellow")
    try:
        from utils import print_v
        print_v(panel)
    except Exception:
        Console().print(panel)


def _log_model_request(model: str, max_tokens: int, num_comps: int) -> None:
    lines = [
        f"model: {model}",
        f"request_timeout_seconds: {DEFAULT_REQUEST_TIMEOUT}",
        f"max_retries: {DEFAULT_MAX_RETRIES}",
        f"retry_max_seconds: {DEFAULT_RETRY_MAX_SECONDS}",
        f"max_tokens: {max_tokens}",
        f"num_comps: {num_comps}",
    ]
    panel = Panel("\n".join(lines), title="Model API request", border_style="cyan")
    try:
        from utils import print_v
        print_v(panel)
    except Exception:
        Console().print(panel)


def _log_embedding_request(model: str, num_inputs: int) -> None:
    lines = [
        f"model: {model}",
        f"request_timeout_seconds: {DEFAULT_REQUEST_TIMEOUT}",
        f"max_retries: {DEFAULT_MAX_RETRIES}",
        f"retry_max_seconds: {DEFAULT_RETRY_MAX_SECONDS}",
        f"num_inputs: {num_inputs}",
    ]
    panel = Panel("\n".join(lines), title="Embedding API request", border_style="cyan")
    try:
        from utils import print_v
        print_v(panel)
    except Exception:
        Console().print(panel)


MODEL_API_RETRY = retry(
    wait=wait_random_exponential(min=1, max=DEFAULT_RETRY_MAX_WAIT),
    stop=(
        stop_after_attempt(DEFAULT_MAX_RETRIES)
        | stop_after_delay(DEFAULT_RETRY_MAX_SECONDS)
    ),
    before_sleep=_log_retry,
    reraise=True,
)


@MODEL_API_RETRY
def gpt_embedding(model: str, texts: List[str]) -> List[List[float]]:
    _log_embedding_request(model, len(texts))
    # 使用 embedding 专用端点（独立于 chat 模型提供商）
    # 例如 chat 用智谱 GLM，embedding 用 OpenAI text-embedding-3-small
    from utils import EMBEDDING_API_KEY, EMBEDDING_API_BASE
    _saved_key = openai.api_key
    _saved_base = openai.api_base
    try:
        openai.api_key = EMBEDDING_API_KEY
        openai.api_base = EMBEDDING_API_BASE
        response = openai.Embedding.create(
            model=model,
            input=texts,
            request_timeout=DEFAULT_REQUEST_TIMEOUT,
        )
    finally:
        openai.api_key = _saved_key
        openai.api_base = _saved_base
    data = response["data"] if isinstance(response, dict) else response.data
    ordered = sorted(
        data,
        key=lambda item: item.get("index") if isinstance(item, dict) else item.index,
    )
    return [
        list(item.get("embedding") if isinstance(item, dict) else item.embedding)
        for item in ordered
    ]


@MODEL_API_RETRY
def gpt_completion(
        model: str,
        prompt: str,
        max_tokens: int = 4096,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
) -> Union[List[str], str]:
    _log_model_request(model, max_tokens, num_comps)
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        n=num_comps,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


@MODEL_API_RETRY
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    num_comps=1,
    extra_body: Optional[dict] = None,
) -> Union[List[str], str]:
    create_kwargs = {}
    if extra_body is not None:
        create_kwargs.update(extra_body)
    _log_model_request(model, max_tokens, num_comps)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        **create_kwargs,
    )
    if num_comps == 1:
        choice = response.choices[0]
        _log_chat_choice(choice)
        return _message_content(_choice_value(choice, "message"))  # type: ignore

    for choice in response.choices:
        _log_chat_choice(choice)
    return [_message_content(_choice_value(choice, "message")) for choice in response.choices]  # type: ignore


class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 4096, temperature: float = 0.2, num_comps: int = 1, extra_body: Optional[dict] = None) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 4096, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 4096, temperature: float = 0.2, num_comps: int = 1, extra_body: Optional[dict] = None) -> Union[List[str], str]:
        return gpt_chat(self.name, messages, max_tokens, temperature, num_comps, extra_body=extra_body)


class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")

class GPT4oMini(GPTChat):
    def __init__(self):
        super().__init__("gpt-4.1")


class GPT35(GPTChat):
    def __init__(self):
        super().__init__("gpt-3.5-turbo")


class GLMChat(GPTChat):
    def __init__(self, model_name: str = "GLM-4.6V-Flash"):
        # GLM uses an OpenAI-compatible ChatCompletion API at the configured base URL.
        super().__init__(model_name)

    def generate_chat(self, messages: List[Message], max_tokens: int = 4096, temperature: float = 0.2, num_comps: int = 1, extra_body: Optional[dict] = None) -> Union[List[str], str]:
        return gpt_chat(
            self.name,
            messages,
            max_tokens,
            temperature,
            num_comps,
            extra_body=extra_body,
        )


class GPTDavinci(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name

    def generate(self, prompt: str, max_tokens: int = 4096, stop_strs: Optional[List[str]] = None, temperature: float = 0, num_comps=1) -> Union[List[str], str]:
        return gpt_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)


class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 4096, temperature: float = 0.2, num_comps: int = 1, extra_body: Optional[dict] = None) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(
                max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )

        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)

        if len(outs) == 1:
            return outs[0]  # type: ignore
        else:
            return outs  # type: ignore

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class StarChat(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/starchat-beta",
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"

        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[:-len("<|end|>")]

        return out


class CodeLlama(HFModelBase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages
        messages = [
            Message(role=messages[1].role, content=self.B_SYS +
                    messages[0].content + self.E_SYS + messages[1].content)
        ] + messages[2:]
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ],
            [],
        )
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        # remove eos token from last message
        messages_tokens = messages_tokens[:-1]
        import torch
        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out
