#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
import time
from importlib.metadata import metadata
from pathlib import Path
from typing import List, Any

import frontmatter
import requests
from jinja2 import Environment, StrictUndefined, Template

environment = Environment(undefined=StrictUndefined, autoescape=True)

TAG_SYSTEM_PROMPT = """
You are a smart assistant specialized in language comprehension. When you receive a
quote, you analyze its meaning and provide a list of 3 to 7 relevant tags that reflect
the subject, theme, or emotional tone of the quote. The tags are short (usually one
word or short phrase, in lowercase), in English.
"""
TAG_SYSTEM_PROMPT_TEMPLATE = environment.from_string(TAG_SYSTEM_PROMPT)
TAG_USER_PROMPT = "What are good tags for this quote? {{ content }}"
TAG_USER_PROMPT_TEMPLATE = environment.from_string(TAG_USER_PROMPT)

OPTIMIZE_TAGS_SYSTEM_PROMPT = """
You are a smart assistant specialized in organizing and optimizing tag lists.
Your task is to analyze long lists of overlapping or semantically similar tags
and reduce them by grouping them under broader, representative tags. Preserve as much
meaning as possible, but aim for fewer, clearer, and more general tags
without redundancy. The tags are short (usually one
word or short phrase, in lowercase), in English
"""
OPTIMIZE_TAGS_SYSTEM_PROMPT_TEMPLATE = environment.from_string(
    OPTIMIZE_TAGS_SYSTEM_PROMPT
)
OPTIMIZE_TAGS_USER_PROMPT = """
I have a long list of tags, many of which overlap or are very specific.
Can you analyze this list and reduce it to a shorter list of broader tags
that summarize the content well? Please group similar or overlapping tags under
a single broader tag where appropriate, while keeping the core meaning intact.
Here's the list: {{ content }}
"""
OPTIMIZE_TAGS_USER_PROMPT_TEMPLATE = environment.from_string(OPTIMIZE_TAGS_USER_PROMPT)

RETAG_SYSTEM_PROMPT = """
You are a smart assistant specialized in language comprehension. When you receive a
quote, you analyze its meaning and provide a list of 3 to 7 relevant tags that reflect
the subject, theme, or emotional tone of the quote. You make pick from this list of tags: {{ optimized_tag_list }}.
"""
RETAG_SYSTEM_PROMPT_TEMPLATE = environment.from_string(RETAG_SYSTEM_PROMPT)
RETAG_USER_PROMPT = "What are good tags for this quote? {{ content }}"
RETAG_USER_PROMPT_TEMPLATE = environment.from_string(RETAG_USER_PROMPT)

def llm_complete(
    sample: dict,
    api_base_url: str,
    api_key: str,
    model: str,
    system: Template,
    prompt: Template,
    output_format: dict,
    verify: bool = True,
) -> dict[str, Any]:
    url = f"{api_base_url}/api/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": system.render(**sample)},
        {"role": "user", "content": prompt.render(**sample)},
    ]

    data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
        "format": output_format,
    }

    response = requests.post(
        url,
        headers=headers,
        json=data,
        verify=verify,
        timeout=60,
    )

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    response_json = response.json()

    for term in [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ]:
        sample[f"ollama_{term}"] = response_json.get(term)

    content = json.loads(response_json.get("message").get("content"))

    for key, value in content.items():
        sample[key] = value

    return sample


def get_optimized_tag_list(md_data, args):
    tag_list = []

    for filename, info in md_data.items():
        tag_list.extend(info["metadata"]["tags"])

    response = llm_complete(
        sample={"content": tag_list},
        api_base_url=args.ollama_base_url,
        api_key=args.ollama_api_key,
        model=args.ollama_model,
        system=OPTIMIZE_TAGS_SYSTEM_PROMPT_TEMPLATE,
        prompt=OPTIMIZE_TAGS_USER_PROMPT_TEMPLATE,
        verify=args.ollama_ssl_verify,
        output_format={
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
        },
    )

    return response["tags"]


def retag_md_data(optimized_tag_list, md_data, args):

    for filename, info in md_data.items():

        response = llm_complete(
            sample={ "content": info["content"], "optimized_tag_list": optimized_tag_list},
            api_base_url=args.ollama_base_url,
            api_key=args.ollama_api_key,
            model=args.ollama_model,
            system=RETAG_SYSTEM_PROMPT_TEMPLATE,
            prompt=RETAG_USER_PROMPT_TEMPLATE,
            verify=args.ollama_ssl_verify,
            output_format={
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tags"],
            },
        )

        md_data[filename]["metadata"]["tags"] = response["tags"]


    return md_data

def get_frontmatter_and_content(directory, args) -> dict[str, dict[str, Any]]:
    data_dict = {}

    for md_file in directory.rglob("*.md"):
        post = frontmatter.load(md_file)

        response = llm_complete(
            sample=post.to_dict(),
            api_base_url=args.ollama_base_url,
            api_key=args.ollama_api_key,
            model=args.ollama_model,
            system=TAG_SYSTEM_PROMPT_TEMPLATE,
            prompt=TAG_USER_PROMPT_TEMPLATE,
            verify=args.ollama_ssl_verify,
            output_format={
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tags"],
            },
        )

        data_dict[md_file] = {"metadata": post.metadata, "content": post.content}
        data_dict[md_file]["metadata"]["tags"] = response["tags"]
    return data_dict


def main():
    parser = argparse.ArgumentParser(description="Searchdata CLI")

    # Required positional argument: input directory
    parser.add_argument(
        "input_directory", help="Path to the directory containing the input files."
    )

    # Verbosity: standaard op WARNING, verhoog naar INFO met --verbose
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Increase output verbosity (default: WARNING).",
    )

    parser.add_argument(
        "--ollama-model",
        default="mistral-small",
        help="The name of the model to use.",
    )

    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11411",
        help="The base URL for Ollama's API. Defaults to http://localhost:11411.",
    )

    parser.add_argument("--ollama-api-key", help="API key to access Ollama's API.")

    parser.add_argument(
        "--ollama-ssl-verify",
        action="store_true",
        help="Whether to verify SSL certificates when connecting to Ollama.",
    )

    args = parser.parse_args()

    # Stel logniveau in
    log_level = logging.INFO if args.verbose else logging.WARNING

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        stream=sys.stderr,
        format="%(asctime)s.%(msecs)03d pid[%(process)d] %(levelname)s %(message)s",
        level=log_level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Starting thematic categorizer...")

    input_path = Path(args.input_directory).resolve()

    if not input_path.exists() or not input_path.is_dir():
        logging.error(f"Invalid input directory: {input_path}")
        sys.exit(1)

    md_data = get_frontmatter_and_content(input_path, args)
    optimized_tag_list = get_optimized_tag_list(md_data, args)
    md_data = retag_md_data(optimized_tag_list, md_data, args)

    for filename, info in md_data.items():
        content = info["content"]
        metadata = info["metadata"]

        post = frontmatter.Post(content=content, **metadata)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))


if __name__ == "__main__":
    main()
