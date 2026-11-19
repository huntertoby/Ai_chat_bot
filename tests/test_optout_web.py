"""D-27 之後唯一保留的上網限制：打過 /不要記我 的人，名字不送到網路上（web_search 的關鍵字、open_url 的網址）。"""
import asyncio
from dataclasses import replace
from urllib.parse import quote

from yuki.brain import OPTOUT_SEARCH_BLOCKED, Brain, web_names_hit
from yuki.config import Settings
from yuki.memory.store import MemoryStore
from yuki.memory.writer import ChatLine
from yuki.rag.decompose import Person

from conftest import FakeEmbedder

AMING = Person("1", "阿明")
XIAOMEI = Person("3", "小美")


class OneToolLLM:
    """假裝模型呼叫一次指定的工具，記下工具回傳。"""

    def __init__(self, tool, args):
        self.tool, self.args, self.results = tool, args, []
        self._slots = asyncio.Semaphore(1)

    async def chat_tools(self, messages, tools, call_tool, **kw):
        self.results.append(await call_tool(self.tool, self.args))
        return "好喔"

    async def chat(self, messages, **kw):
        return "好喔"


def reply_with(tool, args, *, members=(), opted_out=(), optout_names=(), retrieval="agentic"):
    llm = OneToolLLM(tool, args)
    settings = replace(Settings(), retrieval=retrieval, web_enabled=True, decompose_with_llm=False,
                       llm_rerank="off", query_rewrite=False)
    brain = Brain(settings, llm, MemoryStore(":memory:", "fake"), FakeEmbedder(), None)
    brain.opted_out = lambda: set(opted_out)
    brain.optout_names = lambda: set(optout_names)
    sent = []

    async def fake_web(name, a):
        sent.append((name, a))
        return "網路結果"

    brain.web.call = fake_web
    brain.observe("c", ChatLine(AMING, "幫我查"))
    asyncio.run(brain.reply("g", "c", AMING, "幫我查", members=list(members)))
    return sent, llm.results


def test_opted_out_member_name_never_reaches_web_search():
    sent, results = reply_with("web_search", {"query": "小美 學校"}, members=[AMING, XIAOMEI], opted_out={"3"})
    assert sent == [] and results == [OPTOUT_SEARCH_BLOCKED]


def test_members_who_did_not_opt_out_are_searched_normally():
    sent, _ = reply_with("web_search", {"query": "阿明 比賽"}, members=[AMING, XIAOMEI], opted_out={"3"})
    assert sent == [("web_search", {"query": "阿明 比賽"})]


def test_opted_out_name_hidden_in_an_encoded_url_is_blocked():
    url = "https://search.example/?q=" + quote("小美 學校")
    sent, results = reply_with("open_url", {"url": url}, members=[XIAOMEI], opted_out={"3"})
    assert sent == [] and results == [OPTOUT_SEARCH_BLOCKED]
    # 編碼兩次也解得開
    twice = "https://search.example/?q=" + quote(quote("小美"))
    sent, _ = reply_with("open_url", {"url": twice}, members=[XIAOMEI], opted_out={"3"})
    assert sent == []


def test_opted_out_name_blocked_even_when_not_in_this_conversation_member_list():
    """私訊時成員名單是空的：靠 bot 給的「退出的人在各伺服器的名字」擋。"""
    sent, results = reply_with("web_search", {"query": "小美 在哪"}, members=[], optout_names={"小美"})
    assert sent == [] and results == [OPTOUT_SEARCH_BLOCKED]


def test_old_retrieval_path_blocks_opted_out_names_too():
    sent, results = reply_with("open_url", {"url": "https://x.example/小美"}, members=[XIAOMEI],
                               opted_out={"3"}, retrieval="embedding")
    assert sent == [] and results == [OPTOUT_SEARCH_BLOCKED]


def test_only_web_tools_are_checked():
    assert web_names_hit("search_chat", {"keywords": ["小美"]}, {"小美"}) == []
    assert web_names_hit("open_url", {"url": "https://x.example/a+%E5%B0%8F%E7%BE%8E"}, {"小美"}) == ["小美"]
