import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeLLM:
    def __init__(self, json_reply=None, text_reply="好喔"):
        self.json_reply = json_reply
        self.text_reply = text_reply
        self.calls = []

    async def chat_json(self, messages, schema, **kw):
        self.calls.append(messages)
        return self.json_reply

    async def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.text_reply
