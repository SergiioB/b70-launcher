"""Tests for responses_translate — Responses API <-> Chat Completions.

Pure functions only: no sockets, no subprocesses.

Run:  cd b70-launcher && python3 -m unittest tests.test_responses_translate -v
"""
import json
import unittest

try:
    import support_env  # noqa: F401  (puts the repo root on sys.path)
except ImportError:
    from tests import support_env  # noqa: F401

from b70launcher.responses_translate import (
    ChatToResponsesStream,
    chat_completion_to_response,
    responses_to_chat,
)

PATCH = (
    "*** Begin Patch\n"
    "*** Update File: src/app.py\n"
    "@@ def main():\n"
    "-    print(\"hi\")\n"
    "+    print(\"hello, \\\"world\\\"\\n\")\n"
    "*** End Patch\n"
)

APPLY_PATCH_TOOL = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Use the `apply_patch` tool to edit files.",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: begin_patch hunk+ end_patch"},
}
SHELL_TOOL = {
    "type": "function",
    "name": "shell",
    "description": "Runs a shell command.",
    "strict": False,
    "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}},
                   "required": ["command"]},
}


def chunk(delta=None, finish=None, usage=None, choices=True):
    body = {"object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if choices else []}
    if usage is not None:
        body["usage"] = usage
    return body


def run_stream(chunks, tool_types=None):
    stream = ChatToResponsesStream("codex-model", tool_types or {})
    events = stream.start()
    for c in chunks:
        events += stream.feed(c)
    events += stream.finish()
    return stream, events


class TestResponsesToChat(unittest.TestCase):

    def test_multi_turn_parallel_function_calls(self):
        body = {
            "model": "gpt-5-codex",
            "instructions": "You are Codex.",
            "stream": True,
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "list both dirs"}]},
                {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "plan"}],
                 "encrypted_content": None},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Listing."}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_a", "name": "shell",
                 "arguments": "{\"command\":[\"ls\",\"a\"]}"},
                {"type": "function_call", "id": "fc_2", "call_id": "call_b", "name": "shell",
                 "arguments": "{\"command\":[\"ls\",\"b\"]}"},
                {"type": "function_call_output", "call_id": "call_a", "output": "a1\na2"},
                {"type": "function_call_output", "call_id": "call_b",
                 "output": [{"type": "input_text", "text": "b1"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "a has 2, b has 1."}]},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "and this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "original"},
                ]},
            ],
            "tools": [SHELL_TOOL, {"type": "web_search"}, {"type": "local_shell"}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "max_output_tokens": 2048,
            "temperature": 0.2,
            "top_p": 0.9,
            "store": False,
        }
        chat, tool_types = responses_to_chat(body, "ovms-model")
        self.assertEqual(chat["model"], "ovms-model")
        self.assertEqual(tool_types, {"shell": "function"})
        self.assertEqual(chat["messages"], [
            {"role": "system", "content": "You are Codex."},
            {"role": "user", "content": "list both dirs"},
            {"role": "assistant", "content": "Listing.", "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "shell", "arguments": "{\"command\":[\"ls\",\"a\"]}"}},
                {"id": "call_b", "type": "function",
                 "function": {"name": "shell", "arguments": "{\"command\":[\"ls\",\"b\"]}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_a", "content": "a1\na2"},
            {"role": "tool", "tool_call_id": "call_b", "content": "b1"},
            {"role": "assistant", "content": "a has 2, b has 1."},
            {"role": "user", "content": [
                {"type": "text", "text": "and this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}},
            ]},
        ])
        self.assertEqual(chat["tools"], [{"type": "function", "function": {
            "name": "shell", "description": "Runs a shell command.",
            "parameters": SHELL_TOOL["parameters"], "strict": False}}])
        self.assertEqual(chat["tool_choice"], "auto")
        self.assertIs(chat["parallel_tool_calls"], True)
        self.assertEqual(chat["max_tokens"], 2048)
        self.assertEqual((chat["temperature"], chat["top_p"]), (0.2, 0.9))
        self.assertIs(chat["stream"], True)
        self.assertEqual(chat["stream_options"], {"include_usage": True})
        self.assertNotIn("store", chat)
        self.assertNotIn("max_output_tokens", chat)

    def test_tool_calls_without_preceding_text_get_own_assistant_message(self):
        chat, _ = responses_to_chat({"input": [
            {"type": "message", "role": "user", "content": "go"},
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": "{}"},
        ]}, "m")
        self.assertEqual([m["role"] for m in chat["messages"]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(chat["messages"][1]["content"], "")
        self.assertEqual(chat["messages"][3]["tool_calls"][0]["id"], "c2")
        self.assertIs(chat["stream"], False)
        self.assertNotIn("stream_options", chat)

    def test_developer_and_system_messages_merge_into_one_leading_system(self):
        chat, _ = responses_to_chat({
            "instructions": "Base instructions.",
            "input": [
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "Sandbox: workspace-write."}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<env/>"}]},
                {"role": "system", "content": "Late system note."},
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "Part A"}, {"type": "input_text", "text": "Part B"}]},
                {"type": "message", "role": "user", "content": "hi"},
            ],
        }, "m")
        self.assertEqual(chat["messages"], [
            {"role": "system",
             "content": "Base instructions.\n\nSandbox: workspace-write.\n\nLate system note.\n\nPart A\nPart B"},
            {"role": "user", "content": "<env/>"},
            {"role": "user", "content": "hi"},
        ])
        self.assertEqual(sum(m["role"] == "system" for m in chat["messages"]), 1)
        self.assertNotIn("tools", chat)
        self.assertNotIn("tool_choice", chat)

    def test_custom_tool_request_mapping_and_tool_choice(self):
        body = {
            "input": [
                {"type": "message", "role": "user", "content": "fix it"},
                {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_p", "name": "apply_patch",
                 "input": PATCH},
                {"type": "custom_tool_call_output", "call_id": "call_p", "output": "Success. Updated src/app.py"},
            ],
            "tools": [APPLY_PATCH_TOOL, SHELL_TOOL],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
        }
        chat, tool_types = responses_to_chat(body, "m")
        self.assertEqual(tool_types, {"apply_patch": "custom", "shell": "function"})
        call = chat["messages"][1]["tool_calls"][0]
        self.assertEqual(call["id"], "call_p")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"input": PATCH})
        self.assertEqual(chat["messages"][2],
                         {"role": "tool", "tool_call_id": "call_p", "content": "Success. Updated src/app.py"})
        fn = chat["tools"][0]["function"]
        self.assertEqual(fn["name"], "apply_patch")
        self.assertEqual(fn["parameters"]["required"], ["input"])
        self.assertEqual(fn["parameters"]["properties"]["input"]["type"], "string")
        self.assertIn("start: begin_patch hunk+ end_patch", fn["description"])
        self.assertEqual(chat["tool_choice"], {"type": "function", "function": {"name": "apply_patch"}})

    def test_hosted_tool_choice_and_parallel_without_tools_dropped(self):
        chat, tool_types = responses_to_chat({
            "input": "hi", "tools": [{"type": "web_search_preview"}],
            "tool_choice": {"type": "web_search_preview"}, "parallel_tool_calls": False,
        }, "m")
        self.assertEqual(chat["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(tool_types, {})
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            self.assertNotIn(key, chat)


class TestStream(unittest.TestCase):

    def assertSequenced(self, events):
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))

    def test_reasoning_then_text_ordering(self):
        stream, events = run_stream([
            chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning_content": "Think"}),
            chunk({"reasoning": "ing."}),
            chunk({"content": "Hel"}),
            chunk({"content": "lo"}),
            chunk({}, finish="stop"),
        ])
        self.assertSequenced(events)
        self.assertEqual([e["type"] for e in events], [
            "response.created",
            "response.output_item.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ])
        self.assertEqual(events[0]["response"]["status"], "in_progress")
        self.assertEqual(events[4]["text"], "Thinking.")
        self.assertEqual(events[10]["text"], "Hello")
        rs_id, msg_id = events[1]["item"]["id"], events[6]["item"]["id"]
        self.assertNotEqual(rs_id, msg_id)
        self.assertEqual({e["item_id"] for e in events[2:5]}, {rs_id})
        self.assertEqual({e["item_id"] for e in events[7:12]}, {msg_id})
        self.assertEqual((events[1]["output_index"], events[6]["output_index"]), (0, 1))
        response = events[-1]["response"]
        self.assertEqual(response["id"], events[0]["response"]["id"])
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"], [
            {"id": rs_id, "type": "reasoning", "status": "completed",
             "summary": [{"type": "summary_text", "text": "Thinking."}]},
            {"id": msg_id, "type": "message", "status": "completed", "role": "assistant",
             "content": [{"type": "output_text", "text": "Hello", "annotations": []}]},
        ])
        self.assertEqual(response["usage"], {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        })
        self.assertIsNone(stream.usage)

    def test_think_tags_stripped_across_chunk_boundaries(self):
        _, events = run_stream([
            chunk({"content": "<thi"}), chunk({"content": "nk>plan it</th"}),
            chunk({"content": "ink>\n\nAnswer <b>"}), chunk({"content": "bold</b>"}),
        ])
        done = [e for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual([d["item"]["type"] for d in done], ["reasoning", "message"])
        self.assertEqual(done[0]["item"]["summary"][0]["text"], "plan it")
        self.assertEqual(done[1]["item"]["content"][0]["text"], "Answer <b>bold</b>")
        deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        self.assertNotIn("think", deltas)

    def test_usage_chunk_with_empty_choices(self):
        stream, events = run_stream([
            chunk({"content": "ok"}, finish="stop"),
            chunk(choices=False, usage={
                "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 64},
                "completion_tokens_details": {"reasoning_tokens": 12},
            }),
        ])
        self.assertSequenced(events)
        expected = {
            "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 64},
            "output_tokens_details": {"reasoning_tokens": 12},
        }
        self.assertEqual(stream.usage, expected)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["usage"], expected)
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "ok")

    def test_parallel_function_calls_stream(self):
        _, events = run_stream([
            chunk({"content": "Running both."}),
            chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                   "function": {"name": "shell", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"command\":"}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": "[\"ls\"]}"}}]}),
            chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                                   "function": {"name": "shell", "arguments": "{\"command\":[\"pwd\"]}"}}]}),
            chunk({}, finish="tool_calls"),
        ], {"shell": "function"})
        self.assertSequenced(events)
        types = [e["type"] for e in events]
        # The message closes before the first tool item opens.
        self.assertLess(max(i for i, e in enumerate(events) if e["type"] == "response.content_part.done"),
                        types.index("response.function_call_arguments.delta"))
        arg_done = [e for e in events if e["type"] == "response.function_call_arguments.done"]
        self.assertEqual([e["arguments"] for e in arg_done], ["{\"command\":[\"ls\"]}", "{\"command\":[\"pwd\"]}"])
        output = events[-1]["response"]["output"]
        self.assertEqual([o["type"] for o in output], ["message", "function_call", "function_call"])
        self.assertEqual([o["call_id"] for o in output[1:]], ["call_a", "call_b"])
        self.assertEqual([o["arguments"] for o in output[1:]], [e["arguments"] for e in arg_done])
        self.assertEqual(len({o["id"] for o in output}), 3)
        added = [e for e in events if e["type"] == "response.output_item.added"]
        self.assertEqual([e["output_index"] for e in added], [0, 1, 2])
        self.assertEqual(added[1]["item"]["arguments"], "")

    def test_custom_apply_patch_round_trip(self):
        # Request side: the custom tool becomes a function with string `input`.
        chat, tool_types = responses_to_chat({"input": "edit", "tools": [APPLY_PATCH_TOOL]}, "m")
        self.assertEqual(tool_types, {"apply_patch": "custom"})
        # Upstream streams the JSON arguments split at awkward points.
        arguments = json.dumps({"input": PATCH})
        pieces = [arguments[i:i + 7] for i in range(0, len(arguments), 7)]
        chunks = [chunk({"tool_calls": [{"index": 0, "id": "call_patch", "type": "function",
                                         "function": {"name": "apply_patch", "arguments": ""}}]})]
        chunks += [chunk({"tool_calls": [{"index": 0, "function": {"arguments": p}}]}) for p in pieces]
        _, events = run_stream(chunks, tool_types)
        self.assertSequenced(events)
        self.assertEqual([e["type"] for e in events], [
            "response.created", "response.output_item.added", "response.output_item.done",
            "response.completed",
        ])
        added, done = events[1]["item"], events[2]["item"]
        self.assertEqual(added, {"id": added["id"], "type": "custom_tool_call", "status": "in_progress",
                                 "call_id": "call_patch", "name": "apply_patch", "input": ""})
        self.assertEqual(done["type"], "custom_tool_call")
        self.assertEqual(done["id"], added["id"])
        self.assertEqual(done["input"], PATCH)
        self.assertNotIn("arguments", done)
        self.assertEqual(events[-1]["response"]["output"], [done])
        # Next turn: Codex replays the custom call; the patch text survives exactly.
        chat2, _ = responses_to_chat({"input": [
            {"type": "message", "role": "user", "content": "edit"},
            done,
            {"type": "custom_tool_call_output", "call_id": "call_patch", "output": "Done!"},
        ], "tools": [APPLY_PATCH_TOOL]}, "m")
        replayed = chat2["messages"][1]["tool_calls"][0]
        self.assertEqual(replayed["id"], "call_patch")
        self.assertEqual(json.loads(replayed["function"]["arguments"])["input"], PATCH)
        self.assertEqual(chat2["messages"][2], {"role": "tool", "tool_call_id": "call_patch", "content": "Done!"})

    def test_custom_tool_invalid_json_keeps_raw_arguments(self):
        raw = "*** Begin Patch\n*** End Patch"
        _, events = run_stream([chunk({"tool_calls": [{"index": 0, "id": "c", "function": {
            "name": "apply_patch", "arguments": raw}}]})], {"apply_patch": "custom"})
        self.assertEqual(events[-1]["response"]["output"][0]["input"], raw)

    def test_finish_is_idempotent_and_empty_stream_completes(self):
        stream = ChatToResponsesStream("m", {})
        events = stream.start() + stream.finish()
        self.assertEqual([e["type"] for e in events], ["response.created", "response.completed"])
        self.assertEqual(events[-1]["response"]["output"], [])
        self.assertEqual(stream.finish(), [])


class TestNonStream(unittest.TestCase):

    def test_chat_completion_to_response(self):
        payload = {
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1760000000, "model": "ovms",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant",
                "content": "<think>check files</think>\n\nPatching now.",
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "shell", "arguments": "{\"command\":[\"ls\"]}"}},
                    {"id": "call_2", "type": "function",
                     "function": {"name": "apply_patch", "arguments": json.dumps({"input": PATCH})}},
                ],
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        response = chat_completion_to_response(payload, "codex-model",
                                               {"shell": "function", "apply_patch": "custom"})
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["model"], "codex-model")
        self.assertEqual(response["created_at"], 1760000000)
        output = response["output"]
        self.assertEqual([o["type"] for o in output],
                         ["reasoning", "message", "function_call", "custom_tool_call"])
        self.assertEqual(output[0]["summary"], [{"type": "summary_text", "text": "check files"}])
        self.assertEqual(output[1]["content"], [{"type": "output_text", "text": "Patching now.", "annotations": []}])
        self.assertEqual((output[2]["call_id"], output[2]["name"], output[2]["arguments"]),
                         ("call_1", "shell", "{\"command\":[\"ls\"]}"))
        self.assertEqual((output[3]["call_id"], output[3]["input"]), ("call_2", PATCH))
        self.assertEqual(response["usage"], {
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        })

    def test_reasoning_field_and_template_opened_think(self):
        response = chat_completion_to_response({"choices": [{"message": {
            "role": "assistant", "content": "hidden plan</think>Visible."}}]}, "m", {})
        self.assertEqual([o["type"] for o in response["output"]], ["reasoning", "message"])
        self.assertEqual(response["output"][0]["summary"][0]["text"], "hidden plan")
        self.assertEqual(response["output"][1]["content"][0]["text"], "Visible.")
        response = chat_completion_to_response({"choices": [{"message": {
            "role": "assistant", "reasoning_content": "r", "content": "t"}}]}, "m", {})
        self.assertEqual([o["type"] for o in response["output"]], ["reasoning", "message"])
        self.assertEqual(response["usage"]["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
