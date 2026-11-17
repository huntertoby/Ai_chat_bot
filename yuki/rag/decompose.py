"""查詢分解：按「人物」拆成子查詢，每個子查詢之後各自檢索、各自保有名額。

規則優先（零 LLM 呼叫）：
- 提到 2 個以上的人 → 每人一個子查詢
- 只有 1 個人或沒有人 → 不拆，原句 + 1 個改寫合成一組
只有「單一人物但問了多個面向」（例如「Hunter 喜歡的遊戲跟食物」）才呼叫 LLM 拆。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from ..llm import LLMError, LMStudioClient
from .. import prompts
from .timeparse import TimeRange, parse_order, parse_time

log = logging.getLogger(__name__)

_MULTI_ASPECT = re.compile(r"[、,，]|還有|以及|和|跟|與|或是")
_QUESTION = re.compile(r"[?？]|嗎|什麼|哪|誰|多少|幾|如何|怎")
_SELF = re.compile(r"我(?!們)")
_CONNECTOR = re.compile(r"(?:^|\s)(?:跟|和|與|還有|以及|或)(?=\s|$)|[、,，]")


@dataclass(frozen=True)
class Person:
    id: str
    name: str


@dataclass
class SubQuery:
    text: str                          # rerank 用的主要問法
    subject: Person | None = None      # 有值就只搜這個人的記憶
    variants: list[str] = field(default_factory=list)  # 送進 dense / sparse 的所有問法
    time: TimeRange | None = None      # 問題裡的時間說法，日期吻合的記憶加分
    order: str | None = None           # earliest / latest：「最早」「最後一次」類問題，結果依日期排序
    focus: str = ""                    # 給 cross-encoder 的主題（拿掉時間與問法），預設同 text

    def __post_init__(self):
        if not self.variants:
            self.variants = [self.text]
        if not self.focus:
            self.focus = self.text


def find_people(text: str, author: Person, mentions: list[Person], members: list[Person]) -> list[Person]:
    people: dict[str, Person] = {}
    for p in mentions:
        people.setdefault(p.id, p)
    lowered = text.lower()
    for p in members:
        if len(p.name) >= 2 and p.name.lower() in lowered:
            people.setdefault(p.id, p)
    if _SELF.search(text):
        people.setdefault(author.id, author)
    return list(people.values())


def strip_names(text: str, people: list[Person], extra: list[str] = ()) -> str:
    for word in [p.name for p in people] + list(extra):
        text = re.sub(re.escape(word), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[@＠]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ,，、")


def rule_decompose(text: str, author: Person, people: list[Person], wake_words: list[str],
                   max_subqueries: int = 4) -> list[SubQuery]:
    topic = strip_names(text, people, wake_words)
    if len(people) >= 2:
        # 「A 跟 B 都在忙什麼」拿掉人名後會剩「跟 都在忙什麼」，連接詞要一起清掉
        topic_no_self = _CONNECTOR.sub(" ", _SELF.sub("", topic))
        topic_no_self = re.sub(r"\s+", " ", topic_no_self).strip()
        return [SubQuery(f"{p.name} {topic_no_self}".strip(), p) for p in people[:max_subqueries]]
    if len(people) == 1:
        p = people[0]
        resolved = _SELF.sub(p.name if p.id == author.id else author.name, text)
        variants = list(dict.fromkeys([resolved, f"{p.name} {_SELF.sub('', topic)}".strip()]))
        return [SubQuery(resolved, p, variants)]
    return [SubQuery(strip_names(text, [], wake_words) or text)]


# 問「什麼時候」的答案是記憶的日期，不在內文裡；留著這些字，cross-encoder 會判定每筆記憶都答不了
_ASKS_WHEN = re.compile(r"是?在?什麼時候(開始)?|是?哪一?天|幾月幾?[號日]?|是?發生在|最早|第一次|頭一次|最初|一開始"
                        r"|最後一次|最近一次|上一次是?|是不是|有沒有")
_TRAILING = re.compile(r"[嗎呢啊呀吧啦？?！!。，,\s]+$")


def focus_text(text: str, time: TimeRange | None, people: list[Person] = ()) -> str:
    """rerank 用的主題：拿掉時間說法與「什麼時候」類問法。剩下不到兩個字（扣掉人名）就用原句。"""
    focus = text.replace(time.phrase, " ") if time else text
    focus = _TRAILING.sub("", _ASKS_WHEN.sub(" ", focus))
    focus = re.sub(r"\s+", " ", focus).strip()
    rest = strip_names(focus, list(people))
    return focus if len(rest.replace(" ", "")) >= 2 else text


def needs_llm_split(text: str, people: list[Person]) -> bool:
    # 「跟」「和」在閒聊裡很常見，要同時像個問題才值得多花一次 LLM 呼叫
    return (len(people) <= 1 and len(text) >= 8
            and bool(_MULTI_ASPECT.search(text)) and bool(_QUESTION.search(text)))


async def llm_decompose(llm: LMStudioClient, text: str, author: Person, people: list[Person],
                        max_subqueries: int) -> list[SubQuery] | None:
    by_name = {p.name.lower(): p for p in people}
    messages = [
        {"role": "system", "content": prompts.DECOMPOSE_SYSTEM},
        {"role": "user", "content": prompts.decompose_user(text, author.name, [p.name for p in people])},
    ]
    try:
        data = await llm.chat_json(messages, prompts.DECOMPOSE_SCHEMA, name="decompose", max_tokens=400,
                                   reasoning_effort="none")
    except LLMError as e:
        log.warning("查詢分解失敗，改用規則：%s", e)
        return None
    if not data:
        return None
    subs = []
    for q in data.get("queries", [])[:max_subqueries]:
        query = str(q.get("query", "")).strip()
        if not query:
            continue
        person = by_name.get(str(q.get("person", "")).strip().lower())
        if person is None and len(people) == 1:
            person = people[0]
        subs.append(SubQuery(query, person))
    return subs or None


async def decompose(text: str, author: Person, mentions: list[Person], members: list[Person], *,
                    wake_words: list[str], llm: LMStudioClient | None = None,
                    max_subqueries: int = 4, today: date | None = None) -> list[SubQuery]:
    people = find_people(text, author, mentions, members)
    subs = None
    if llm is not None and needs_llm_split(text, people):
        subs = await llm_decompose(llm, text, author, people, max_subqueries)
    subs = subs or rule_decompose(text, author, people, wake_words, max_subqueries)
    # 時間說法套用到整句：「Hunter 跟 Toby 上週在幹嘛」兩個子查詢都是上週
    time = parse_time(text, today or date.today())
    order = parse_order(text)
    for sq in subs:
        sq.time, sq.order = time, order
        sq.focus = focus_text(sq.text, time, people)
    return subs
