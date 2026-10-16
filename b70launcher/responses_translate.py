"""Pure translation between the OpenAI Responses API and Chat Completions.

Codex speaks the Responses API (`POST /v1/responses`); some engines (OpenVINO
OVMS) only serve `/v1/chat/completions`. The gateway uses:

* `responses_to_chat` to turn a Responses request into a chat request;
* `ChatToResponsesStream` to turn parsed `chat.completion.chunk` dicts into
  Responses stream event dicts (the caller frames them as SSE);
* `chat_completion_to_response` for non-streaming replies.

No I/O; stdlib only. Freeform (`custom`) tools such as Codex's `apply_patch`
are exposed upstream as a function with one string parameter `input` and
mapped back to `custom_tool_call` items. `<think>...</think>` spans in text
are removed from the message and surfaced as reasoning.
"""
from __future__ import annotations

import json
import time
import uuid

_TEXT_PART_TYPES = ("input_text", "output_text", "text")
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_IMAGE_DETAILS = ("low", "high", "auto")


# ---------------------------------------------------------------------------
# Request: Responses -> Chat Completions
# ---------------------------------------------------------------------------

def _content_parts(content, images: bool) -> list:
    """Responses content (str, part dict or list of parts) -> chat parts."""
    if content is None:
        return []
    if isinstance(content, (str, dict)):
        content = [content]
    if not isinstance(content, list):
        return []
    parts = []
    for part in content:
        if isinstance(part, str):
            if part:
                parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in _TEXT_PART_TYPES:
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
        elif kind == "input_image" and images:
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str) or not url:
                continue
            image = {"url": url}
            detail = part.get("detail")
            if detail == "original":
                detail = "high"
            if detail in _IMAGE_DETAILS:
                image["detail"] = detail
            parts.append({"type": "image_url", "image_url": image})
    return parts


def _parts_text(parts: list) -> str:
    return "\n".join(p["text"] for p in parts if p["type"] == "text")


def _chat_content(parts: list):
    """String when text-only (widest template support), parts list otherwise."""
    if any(p["type"] == "image_url" for p in parts):
        return parts
    return _parts_text(parts)


def _attach_tool_call(messages: list, call: dict) -> None:
    """Coalesce into the preceding assistant message (text or tool calls)."""
    last = messages[-1] if messages else None
    if last is not None and last["role"] == "assistant":
        last.setdefault("tool_calls", []).append(call)
    else:
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})


def _tool_output(messages: list, images: list, item: dict) -> None:
    parts = _content_parts(item.get("output"), images=True)
    images.extend(p for p in parts if p["type"] == "image_url")
    messages.append({
        "role": "tool",
        "tool_call_id": str(item.get("call_id") or ""),
        "content": _parts_text(parts),
    })


def _flush_tool_images(messages: list, images: list) -> None:
    """Chat `tool` messages carry text only; images follow as one user turn."""
    if images:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Images returned by the tool call(s) above:"}] + images,
        })
        images.clear()


def _input_to_messages(value, system_texts: list) -> list:
    if isinstance(value, str):
        return [{"role": "user", "content": value}] if value else []
    if not isinstance(value, list):
        return []
    messages = []
    images = []
    for item in value:
        if isinstance(item, str):
            item = {"type": "message", "role": "user", "content": item}
        if not isinstance(item, dict):
            continue
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind not in ("function_call_output", "custom_tool_call_output"):
            _flush_tool_images(messages, images)
        if kind == "message":
            role = item.get("role")
            if role in ("system", "developer"):
                text = _parts_text(_content_parts(item.get("content"), images=False))
                if text:
                    system_texts.append(text)
            elif role == "user":
                parts = _content_parts(item.get("content"), images=True)
                if parts:
                    messages.append({"role": "user", "content": _chat_content(parts)})
            elif role == "assistant":
                text = _parts_text(_content_parts(item.get("content"), images=False))
                if text:
                    messages.append({"role": "assistant", "content": text})
        elif kind == "function_call":
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments if arguments is not None else {})
            _attach_tool_call(messages, {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "type": "function",
                "function": {"name": str(item.get("name") or ""), "arguments": arguments},
            })
        elif kind == "custom_tool_call":
            raw = item.get("input")
            _attach_tool_call(messages, {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": json.dumps({"input": raw if isinstance(raw, str) else ""}),
                },
            })
        elif kind in ("function_call_output", "custom_tool_call_output"):
            _tool_output(messages, images, item)
        # reasoning, hosted-tool calls and unknown items are not replayed.
    _flush_tool_images(messages, images)
    return messages


def _custom_tool_description(tool: dict) -> str:
    text = str(tool.get("description") or "").strip()
    note = "Pass the complete raw tool input as the string argument `input`."
    fmt = tool.get("format")
    if isinstance(fmt, dict) and isinstance(fmt.get("definition"), str) and fmt["definition"].strip():
        note += (f" `input` must match this {fmt.get('syntax') or 'grammar'} "
                 f"grammar:\n{fmt['definition'].strip()}")
    return f"{text}\n\n{note}" if text else note


def _convert_tools(tools) -> tuple:
    chat_tools = []
    tool_types = {}
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        kind = tool.get("type")
        if kind == "function":
            spec = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = spec.get("name")
            if not isinstance(name, str) or not name:
                continue
            fn = {
                "name": name,
                "description": str(spec.get("description") or ""),
                "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
            }
            if isinstance(spec.get("strict"), bool):
                fn["strict"] = spec["strict"]
            tool_types[name] = "function"
        elif kind == "custom":
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            fn = {
                "name": name,
                "description": _custom_tool_description(tool),
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": "Raw tool input."}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            }
            tool_types[name] = "custom"
        else:  # hosted tools (web_search, local_shell, file_search, mcp, ...)
            continue
        chat_tools.append({"type": "function", "function": fn})
    return chat_tools, tool_types


def _convert_tool_choice(choice, tool_types: dict):
    if choice in ("auto", "none", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") in ("function", "custom"):
        name = choice.get("name")
        if not name and isinstance(choice.get("function"), dict):
            name = choice["function"].get("name")
        if name in tool_types:
            return {"type": "function", "function": {"name": name}}
    return None  # hosted-tool or unknown choice: let the engine default apply


def responses_to_chat(body: dict, model: str) -> tuple:
    """Responses request -> (chat request, tool_types).

    `instructions` plus every developer/system message become ONE leading
    system message. `tool_types` maps each forwarded tool name to
    'function' or 'custom'; pass it to the response translators.
    """
    system_texts = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        system_texts.append(instructions)
    messages = _input_to_messages(body.get("input"), system_texts)
    if system_texts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_texts)})
    if not messages:
        messages = [{"role": "user", "content": ""}]

    chat = {"model": model, "messages": messages}
    tools, tool_types = _convert_tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
        choice = _convert_tool_choice(body.get("tool_choice"), tool_types)
        if choice is not None:
            chat["tool_choice"] = choice
        if isinstance(body.get("parallel_tool_calls"), bool):
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            chat[key] = body[key]
    if body.get("stream"):
        chat["stream"] = True
        chat["stream_options"] = {"include_usage": True}
    else:
        chat["stream"] = False
    return chat, tool_types


# ---------------------------------------------------------------------------
# Response: Chat Completions -> Responses
# ---------------------------------------------------------------------------

def _token(*values) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _normalize_usage(usage: dict) -> dict:
    inp = _token(usage.get("prompt_tokens"), usage.get("input_tokens")) or 0
    out = _token(usage.get("completion_tokens"), usage.get("output_tokens")) or 0
    total = _token(usage.get("total_tokens"))
    in_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    out_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    cached = _token(in_details.get("cached_tokens")) if isinstance(in_details, dict) else None
    reasoning = _token(out_details.get("reasoning_tokens")) if isinstance(out_details, dict) else None
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out if total is None else total,
        "input_tokens_details": {"cached_tokens": cached or 0},
        "output_tokens_details": {"reasoning_tokens": reasoning or 0},
    }


def _partial_tag(text: str, tags: tuple) -> int:
    """Length of the longest suffix of `text` that could start one of `tags`."""
    for size in range(min(len(text), max(len(t) for t in tags) - 1), 0, -1):
        tail = text[-size:]
        if any(t.startswith(tail) for t in tags):
            return size
    return 0


def _custom_input(arguments: str) -> str:
    """Function-call arguments of a custom tool -> raw `input` text."""
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return arguments
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("input"), str):
            return parsed["input"]
        values = list(parsed.values())
        if len(values) == 1 and isinstance(values[0], str):
            return values[0]
    return arguments


class ChatToResponsesStream:
    """Parsed chat.completion.chunk dicts -> Responses stream event dicts.

    Call `start()` once, `feed()` per chunk, `finish()` once. Every returned
    event carries a `sequence_number` counting from 0. Only one item streams
    at a time: opening reasoning, text or a tool call closes the item before
    it, so `output_index` order equals emission order.
    """

    def __init__(self, model: str, tool_types: dict, created_at: int | None = None):
        self.model = model
        self.tool_types = dict(tool_types or {})
        self.usage: dict | None = None
        token = uuid.uuid4().hex
        self.response_id = f"resp_{token}"
        self._tag = token[:16]
        self.created_at = int(time.time()) if created_at is None else int(created_at)
        self._seq = 0
        self._items = []        # item states in output order
        self._reasoning = None  # open reasoning state
        self._message = None    # open message state
        self._tools = {}        # upstream tool-call index -> state
        self._think = False     # inside <think> in the content stream
        self._hold = ""         # content tail that may be a split think tag
        self._done = False

    # -- public -----------------------------------------------------------

    def start(self) -> list:
        return [self._event("response.created", response=self._response("in_progress"))]

    def feed(self, chunk: dict) -> list:
        out = []
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = _normalize_usage(usage)
        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict):
            return out
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            self._reasoning_delta(out, reasoning)
        content = delta.get("content")
        if isinstance(content, str) and content:
            self._content(out, content)
        for call in delta.get("tool_calls") or []:
            if isinstance(call, dict):
                self._tool_delta(out, call)
        return out

    def finish(self) -> list:
        if self._done:
            return []
        self._done = True
        out = []
        if self._hold:
            held, self._hold = self._hold, ""
            (self._reasoning_delta if self._think else self._text_delta)(out, held)
        self._close_reasoning(out)
        self._close_message(out)
        self._close_tools(out)
        for state in self._tools.values():
            if not state["opened"]:
                self._open_tool(out, state)
        self._close_tools(out)
        out.append(self._event("response.completed", response=self._response("completed")))
        return out

    # -- events -----------------------------------------------------------

    def _event(self, kind: str, **fields) -> dict:
        event = {"type": kind, "sequence_number": self._seq}
        event.update(fields)
        self._seq += 1
        return event

    def _response(self, status: str) -> dict:
        final = status == "completed"
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": [self._item(s, "completed") for s in self._items] if final else [],
            "usage": (self.usage or _normalize_usage({})) if final else None,
            "error": None,
            "incomplete_details": None,
        }

    def _item(self, state: dict, status: str) -> dict:
        kind = state["kind"]
        if kind == "reasoning":
            text = state["text"]
            return {"id": state["id"], "type": "reasoning", "status": status,
                    "summary": [{"type": "summary_text", "text": text}] if text else []}
        if kind == "message":
            text = state["text"]
            return {"id": state["id"], "type": "message", "status": status, "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}] if text else []}
        item = {"id": state["id"], "type": kind, "status": status,
                "call_id": state["call_id"], "name": state["name"]}
        final = status == "completed"
        if kind == "custom_tool_call":
            item["input"] = _custom_input(state["arguments"]) if final else ""
        else:
            item["arguments"] = (state["arguments"] or "{}") if final else ""
        return item

    def _open(self, out: list, kind: str, prefix: str, **extra) -> dict:
        index = len(self._items)
        state = {"kind": kind, "id": f"{prefix}_{self._tag}_{index}", "output_index": index}
        state.update(extra)
        self._items.append(state)
        out.append(self._event("response.output_item.added", output_index=index,
                               item=self._item(state, "in_progress")))
        return state

    # -- reasoning --------------------------------------------------------

    def _reasoning_delta(self, out: list, text: str) -> None:
        if self._reasoning is None:
            text = text.lstrip()
            if not text:
                return
            self._close_message(out)
            self._close_tools(out)
            self._reasoning = self._open(out, "reasoning", "rs", text="")
        state = self._reasoning
        state["text"] += text
        out.append(self._event("response.reasoning_summary_text.delta", item_id=state["id"],
                               output_index=state["output_index"], summary_index=0, delta=text))

    def _close_reasoning(self, out: list) -> None:
        state, self._reasoning = self._reasoning, None
        if state is None:
            return
        out.append(self._event("response.reasoning_summary_text.done", item_id=state["id"],
                               output_index=state["output_index"], summary_index=0,
                               text=state["text"]))
        out.append(self._event("response.output_item.done", output_index=state["output_index"],
                               item=self._item(state, "completed")))

    # -- message text -----------------------------------------------------

    def _content(self, out: list, text: str) -> None:
        """Split content on <think> tags, holding back a possibly split tag."""
        buf, self._hold = self._hold + text, ""
        while buf:
            if self._think:
                at = buf.find(_THINK_CLOSE)
                if at >= 0:
                    self._reasoning_delta(out, buf[:at])
                    buf, self._think = buf[at + len(_THINK_CLOSE):], False
                    continue
                keep = _partial_tag(buf, (_THINK_CLOSE,))
            else:
                hits = [(buf.find(tag), tag) for tag in (_THINK_OPEN, _THINK_CLOSE)]
                hits = [hit for hit in hits if hit[0] >= 0]
                if hits:
                    at, tag = min(hits)
                    self._text_delta(out, buf[:at])
                    buf, self._think = buf[at + len(tag):], tag == _THINK_OPEN
                    continue
                keep = _partial_tag(buf, (_THINK_OPEN, _THINK_CLOSE))
            emit, self._hold = buf[:len(buf) - keep], buf[len(buf) - keep:]
            if emit:
                (self._reasoning_delta if self._think else self._text_delta)(out, emit)
            return

    def _text_delta(self, out: list, text: str) -> None:
        if self._message is None:
            text = text.lstrip()
            if not text:
                return
            self._close_reasoning(out)
            self._close_tools(out)
            state = self._open(out, "message", "msg", text="")
            out.append(self._event("response.content_part.added", item_id=state["id"],
                                   output_index=state["output_index"], content_index=0,
                                   part={"type": "output_text", "text": "", "annotations": []}))
            self._message = state
        state = self._message
        state["text"] += text
        out.append(self._event("response.output_text.delta", item_id=state["id"],
                               output_index=state["output_index"], content_index=0, delta=text))

    def _close_message(self, out: list) -> None:
        state, self._message = self._message, None
        if state is None:
            return
        text = state["text"]
        out.append(self._event("response.output_text.done", item_id=state["id"],
                               output_index=state["output_index"], content_index=0, text=text))
        out.append(self._event("response.content_part.done", item_id=state["id"],
                               output_index=state["output_index"], content_index=0,
                               part={"type": "output_text", "text": text, "annotations": []}))
        out.append(self._event("response.output_item.done", output_index=state["output_index"],
                               item=self._item(state, "completed")))

    # -- tool calls -------------------------------------------------------

    def _tool_key(self, call: dict):
        index = call.get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            return index
        call_id = call.get("id")
        for key, state in self._tools.items():
            if call_id and state["upstream_id"] == call_id:
                return key
        if call_id or not self._tools:
            return ("n", len(self._tools))
        return list(self._tools)[-1]

    def _tool_delta(self, out: list, call: dict) -> None:
        key = self._tool_key(call)
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        state = self._tools.get(key)
        if state is None:
            state = {"upstream_id": "", "name": "", "arguments": "", "opened": False, "closed": False}
            self._tools[key] = state
        if not state["upstream_id"] and isinstance(call.get("id"), str):
            state["upstream_id"] = call["id"]
        name = fn.get("name")
        if not state["opened"] and isinstance(name, str) and name:
            state["name"] = name  # some servers repeat the full name per chunk
        args = fn.get("arguments")
        if args is not None and not isinstance(args, str):
            args = json.dumps(args)
        if args:
            state["arguments"] += args
        if not state["opened"]:
            if state["name"]:
                self._open_tool(out, state)  # flushes arguments buffered so far
        elif args and not state["closed"] and state["kind"] == "function_call":
            out.append(self._event("response.function_call_arguments.delta", item_id=state["id"],
                                   output_index=state["output_index"], delta=args))

    def _open_tool(self, out: list, state: dict) -> None:
        self._close_reasoning(out)
        self._close_message(out)
        self._close_tools(out)
        custom = self.tool_types.get(state["name"]) == "custom"
        index = len(self._items)
        state.update(
            kind="custom_tool_call" if custom else "function_call",
            id=f"{'ctc' if custom else 'fc'}_{self._tag}_{index}",
            output_index=index,
            call_id=state["upstream_id"] or f"call_{self._tag}_{index}",
            opened=True,
        )
        self._items.append(state)
        out.append(self._event("response.output_item.added", output_index=index,
                               item=self._item(state, "in_progress")))
        if state["arguments"] and not custom:
            out.append(self._event("response.function_call_arguments.delta", item_id=state["id"],
                                   output_index=index, delta=state["arguments"]))

    def _close_tools(self, out: list) -> None:
        for state in self._tools.values():
            if not state["opened"] or state["closed"]:
                continue
            state["closed"] = True
            item = self._item(state, "completed")
            if state["kind"] == "function_call":
                out.append(self._event("response.function_call_arguments.done", item_id=state["id"],
                                       output_index=state["output_index"],
                                       arguments=item["arguments"]))
            out.append(self._event("response.output_item.done", output_index=state["output_index"],
                                   item=item))


def chat_completion_to_response(payload: dict, model: str, tool_types: dict) -> dict:
    """Non-streaming chat.completion -> Responses `response` object."""
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        content = _parts_text(_content_parts(content, images=False))
    if isinstance(content, str) and _THINK_CLOSE in content \
            and _THINK_OPEN not in content.split(_THINK_CLOSE, 1)[0]:
        content = _THINK_OPEN + content  # template opened <think> in the prompt
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        reasoning = message.get("reasoning")
    calls = [dict(call, index=i) for i, call in enumerate(message.get("tool_calls") or [])
             if isinstance(call, dict)]
    stream = ChatToResponsesStream(model, tool_types, created_at=_token(payload.get("created")))
    stream.feed({
        "choices": [{"delta": {"reasoning_content": reasoning, "content": content, "tool_calls": calls}}],
        "usage": payload.get("usage"),
    })
    return stream.finish()[-1]["response"]
