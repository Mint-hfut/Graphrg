import asyncio
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union
import numpy as np
import tiktoken
from half_json.core import JSONFixer
logger = logging.getLogger("nano-graphrag")
ENCODER = None
from json_repair import repair_json


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    try:
        maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
        if maybe_json_str is not None:
            return repair_json(maybe_json_str.group(0), ensure_ascii=False)
        else:
            jsons = remove_after_last_pattern(content)+"}"
            return locate_json_string_body_from_string(jsons)
    except Exception as e:
        jsons = remove_after_last_pattern(content)
        return locate_json_string_body_from_string(jsons)


def remove_after_last_pattern(s, pattern='},'):
    text = s[0]
    last_pos = text.rfind(pattern)
    if last_pos == -1:
        return s
    else:
        return text[:last_pos+1] + "]}"

def fix_json_bracket(json_str):
    stack = []
    missing = []
    for i, char in enumerate(json_str):
        if char == '{' and i>0 and i<len(json_str)-1 and len(stack)==0:
            stack.append((char, i))
        elif char == '}' and  i>0 and i<len(json_str)-1 and len(stack)>0:
            stack.pop()
        elif char =='{' and i>0 and i<len(json_str)-1 and len(stack)>0:
            missing.append(("}", i))
    if stack:
        last_quotation_index = json_str.rfind('"')
        json_str = json_str[:last_quotation_index+1] + "}" + json_str[last_quotation_index+1:]
    for bracket, index in reversed(missing):
        comma_index = json_str[:index].rfind(',')
        if json_str[comma_index+1:index].isspace():
            index = comma_index
        json_str = json_str[:index] + bracket + json_str[index:]
    f = JSONFixer()
    json_str = f.fix(json_str).line
    return json_str

def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except :
        try:
            print("try to fix the json")
            json_str = fix_json_bracket(json_str)
            data = json.loads(json_str,strict=False)
            return data
        except json.JSONDecodeError as e:
            try:
                if "Unterminated" in e.msg:
                    if json_str[e.pos + 1] in ["]","}"]:
                        json_str = json_str[:e.pos] + json_str[e.pos+1:]
                        data = json.loads(json_str,strict=False)
                        return data
                    brace_index = json_str.find('}', e.pos + 1)
                    if brace_index != -1 :
                        json_str = json_str[:brace_index] + '"' + json_str[brace_index:]
                data = json.loads(json_str,strict=False)
                return data
            except:
                logger.error(f"Failed to parse JSON: {json_str}")



def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name) as f:
        return json.load(f)


# it's dirty to type, so it's a good way to have fun
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [",\t".join([str(data_dd) for data_dd in data_d]) for data_d in data]
    )


# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


# Utils types -----------------------------------------------------------------------
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


# Decorators ------------------------------------------------------------------------
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro
