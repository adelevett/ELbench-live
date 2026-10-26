import sqlite3
from typing import Tuple, Dict, Any
from pathlib import Path
import json
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.base import Checkpoint


def _to_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _stringify_block_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text not in (None, ""):
                    parts.append(str(text))
                continue
            if item not in (None, ""):
                parts.append(str(item))
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        text = value.get("text")
        if text not in (None, ""):
            return str(text)
        return json.dumps(_to_jsonable(value), ensure_ascii=False)
    return str(value)


def _split_reasoning_and_response(content: Any) -> tuple[str, str]:
    if isinstance(content, str):
        if "</think>" in content:
            content_split = content.split("</think>", 1)
            return content_split[0].strip(), content_split[1].strip()
        return "", content.strip()

    if isinstance(content, list):
        reasoning_parts: list[str] = []
        response_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = str(block.get("type", "")).strip().lower()
                if block_type == "reasoning":
                    reasoning_text = _stringify_block_text(block.get("summary"))
                    if not reasoning_text:
                        reasoning_text = _stringify_block_text(block.get("text"))
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                    continue
                if block_type in {"text", "output_text"}:
                    text = _stringify_block_text(block.get("text"))
                    if text:
                        response_parts.append(text)
                    continue
                fallback = _stringify_block_text(block)
                if fallback:
                    response_parts.append(fallback)
                continue

            fallback = _stringify_block_text(block)
            if fallback:
                response_parts.append(fallback)
        return "\n".join(reasoning_parts).strip(), "\n".join(response_parts).strip()

    response = _stringify_block_text(content).strip()
    return "", response


async def aexport_json(input_path: Path) -> Tuple[Path, Dict[str, Any]]:
    conn = sqlite3.connect(input_path)
    cursor = conn.cursor()
    sql = "select checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint from checkpoints"
    cursor.execute(sql)
    results = cursor.fetchall()[-1]
    cns, cid, pcid, c = results
    jps = JsonPlusSerializer()
    checkpoint: Checkpoint = jps.loads_typed(("msgpack", c))
    messages = []
    for m in checkpoint.get("channel_values")["messages"]:
        if m.name is None:
            continue
        reasoning, response = _split_reasoning_and_response(getattr(m, "content", None))
        item = {"role": m.name, "content": response, "reasoning": reasoning}
        usage_metadata = getattr(m, "usage_metadata", None)
        response_metadata = getattr(m, "response_metadata", None)
        additional_kwargs = getattr(m, "additional_kwargs", None)
        if usage_metadata not in (None, {}):
            item["usage_metadata"] = _to_jsonable(usage_metadata)
        if response_metadata not in (None, {}):
            item["response_metadata"] = _to_jsonable(response_metadata)
        if additional_kwargs not in (None, {}):
            item["additional_kwargs"] = _to_jsonable(additional_kwargs)
        messages.append(item)

    sql = "select key, value from task"
    cursor.execute(sql)
    results = cursor.fetchall()
    obj = {"task": {}, "messages": messages}

    for key, value in results:
        obj["task"][key] = value

    return input_path, obj
