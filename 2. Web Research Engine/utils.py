import json
import re


def parse_llm_json(message_content: str) -> dict:
    match = re.search(r"[\{\[].*[\}\]]", message_content, re.DOTALL)
    json_str = match.group(0) if match else message_content
    return json.loads(json_str)
