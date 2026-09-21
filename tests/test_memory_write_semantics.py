"""2026-09-19 用户口径：记忆写入语义按"谁发起"分流。

决策（用户回 c a a）：
  M1=c  判定"用户明确要求记住"= 运行时正则（权威）+ 模型 explicit 参数（补充）
  M2=a  "只问一次"落在对话层：模型推断出的偏好先问一句，用户同意后再 note
  M3=a  显式直写也要留痕（候选生来就是 confirmed，可在记忆页删）

新增需求：模型自己推断的偏好**不每轮试探**，改为每 N 个用户轮回顾一次
（``memory_review_every_turns``，默认 10，0 = 关闭）。

本文件只测确定性部分（正则、分流、节奏、落地、留痕），真模型端到端在
tmp/qa-sweep-20260919/ 的 CDP/acc 脚本里。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.collections import CollectionService
from src.agent.memory.service import MemoryService
from src.agent.runtime import (
    MEMORY_CONTRACT,
    MEMORY_REVIEW_INSTRUCTION,
    PROMPT_SNAPSHOT_VERSION,
    REPLY_CONTRACT_REMINDER,
    _is_explicit_memory_request,
)
from src.agent.tools import Observation, ToolRunner
from src.config.settings import AgentSettings
from src.llm.types import ToolCall
from src.storage.agent_db import AgentDB


class _EmptyRegistry:
    """CollectionService 的 registry 只需要列文档；记忆路径根本不碰资料集。"""

    def list_documents(self) -> list:
        return []


# --------------------------------------------------------------------------- #
# 1) 正则：只认祈使/宣告式说法，不能把"我记得论文里…"当成指令
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "记住这个要求:我询问图表相关内容时,将图表原样展示出来",
        "记下我的偏好：先给直觉再给公式",
        "帮我记住：我在准备 AI 应用开发实习",
        "以后都用中文回答",
        "之后都先给结论再给推导",
        "别忘了：我 2027 年毕业",
        "每次都把表格原样展示",
        "remember this: I prefer Chinese answers",
        "always answer in Chinese",
    ],
)
def test_explicit_memory_request_detected(text: str) -> None:
    assert _is_explicit_memory_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "我记得论文里提到过 Table 4",
        "这个我记不住",
        "记不住也没关系",
        "请讲解一下第 3 节",
        "你记得我说过什么吗",
        "table 4 的数字是多少",
        "",
    ],
)
def test_non_requests_are_not_explicit(text: str) -> None:
    """假阳性会把没让记的东西直写长期库 —— 这比漏判更贵（M3 只能靠删除回退）。"""

    assert _is_explicit_memory_request(text) is False


# --------------------------------------------------------------------------- #
# 2) 分流：显式 → 直写长期库 + 留痕；推断 → 仍然 pending
# --------------------------------------------------------------------------- #

def _db(tmp_path) -> AgentDB:
    # AgentDB 在构造里就跑迁移，没有单独的 initialize()
    return AgentDB(tmp_path / "agent.db")


def _memory(db: AgentDB) -> MemoryService:
    return MemoryService(
        db=db,
        collections=CollectionService(db=db, registry=_EmptyRegistry()),
        settings=SimpleNamespace(digest_chars=600),
    )


def test_explicit_note_writes_through_and_stays_auditable(tmp_path) -> None:
    db = _db(tmp_path)
    memory = _memory(db)
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="deep")

    result = memory.note_confirmed(session_id=session_id, kind="preference", key="chart", value="原样展示")

    assert result["written"] is True
    # 长期库立刻生效
    assert db.get_preferences().get("chart") == "原样展示"
    # 仍然留痕，且是 confirmed（记忆管理页可查可删）
    rows = db.list_candidates(session_id, status="confirmed")
    assert [row["id"] for row in rows] == [result["candidate_id"]]
    assert db.list_candidates(session_id, status="pending") == []


def test_inferred_note_stays_pending(tmp_path) -> None:
    db = _db(tmp_path)
    memory = _memory(db)
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="deep")

    candidate_id = memory.note(session_id=session_id, kind="preference", key="style", value="先直觉")

    assert db.get_preferences().get("style") is None
    assert [row["id"] for row in db.list_candidates(session_id, status="pending")] == [candidate_id]


def test_progress_without_collection_does_not_pretend_to_land(tmp_path) -> None:
    """progress 需要绑定资料集才落得下去 —— 落不下去就留在 pending，不许假装成功。"""

    db = _db(tmp_path)
    memory = _memory(db)
    session_id = db.create_session(collection_id=None, mode="deep")

    result = memory.note_confirmed(session_id=session_id, kind="progress", key="last_focus", value="attention")

    assert result["written"] is False
    assert result["reason"] == "progress_needs_collection"
    assert len(db.list_candidates(session_id, status="pending")) == 1


def test_progress_with_collection_lands(tmp_path) -> None:
    db = _db(tmp_path)
    memory = _memory(db)
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="deep")

    result = memory.note_confirmed(
        session_id=session_id,
        kind="progress",
        key="last_focus",
        value="attention",
        collection_id=collection_id,
    )

    assert result["written"] is True
    assert db.get_progress(collection_id).get("last_focus") == "attention"


# --------------------------------------------------------------------------- #
# 3) 工具层：显式标志 → 直写并如实告知模型；否则 pending
# --------------------------------------------------------------------------- #

def _runner(db: AgentDB, tmp_path) -> ToolRunner:
    return ToolRunner(
        adapter=object(),
        collections=CollectionService(db=db, registry=_EmptyRegistry()),
        memory=_memory(db),
        settings=AgentSettings(db_path=str(tmp_path / "agent.db")),
    )


def _note_call(**arguments) -> ToolCall:
    return ToolCall(id="call_1", name="memory_note", arguments=arguments)


def test_tool_auto_writes_on_explicit_flag(tmp_path) -> None:
    db = _db(tmp_path)
    runner = _runner(db, tmp_path)
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="deep")

    observation = runner.run(
        _note_call(kind="preference", key="chart", value="原样展示"),
        collection_id=collection_id,
        session_id=session_id,
        explicit_memory=True,
    )

    assert isinstance(observation, Observation)
    assert observation.memory_auto_written is True
    assert "saved to long-term memory" in observation.content
    assert db.get_preferences().get("chart") == "原样展示"


def test_tool_auto_writes_on_model_flag(tmp_path) -> None:
    """M1=c 的补充半边：正则没命中、但模型自己标了 explicit，也要直写。"""

    db = _db(tmp_path)
    runner = _runner(db, tmp_path)
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="deep")

    observation = runner.run(
        _note_call(kind="preference", key="chart", value="原样展示", explicit=True),
        collection_id=collection_id,
        session_id=session_id,
        explicit_memory=False,
    )

    assert observation.memory_auto_written is True
    assert db.get_preferences().get("chart") == "原样展示"


def test_tool_stays_pending_without_explicit(tmp_path) -> None:
    db = _db(tmp_path)
    runner = _runner(db, tmp_path)
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="deep")

    observation = runner.run(
        _note_call(kind="preference", key="style", value="先直觉"),
        collection_id=collection_id,
        session_id=session_id,
    )

    assert observation.memory_auto_written is False
    assert "pending candidate" in observation.content
    assert db.get_preferences().get("style") is None


# --------------------------------------------------------------------------- #
# 4) 节奏：每 N 个**用户轮**回顾一次，工具轮不算
# --------------------------------------------------------------------------- #

def _runtime(db: AgentDB, tmp_path, every: int):
    from src.agent.runtime import AgentRuntime

    class _LLM:
        pass

    return AgentRuntime(
        db=db,
        llm=_LLM(),
        tools=_runner(db, tmp_path),
        settings=AgentSettings(db_path=str(tmp_path / "agent.db"), memory_review_every_turns=every),
    )


@pytest.mark.parametrize(
    "user_turns,every,expected",
    [
        (1, 10, False),
        (9, 10, False),
        (10, 10, True),
        (20, 10, True),
        (5, 5, True),
        (7, 5, False),
        (3, 0, False),   # 0 = 关闭定期回顾
    ],
)
def test_memory_review_cadence(tmp_path, user_turns: int, every: int, expected: bool) -> None:
    db = _db(tmp_path)
    runtime = _runtime(db, tmp_path, every)
    collection_id = db.upsert_collection("papers")
    session_id = db.create_session(collection_id=collection_id, mode="deep")

    for index in range(user_turns):
        db.append_message(session_id, "user", f"问题 {index + 1}")
        # 工具轮不该推进节奏
        db.append_message(session_id, "assistant", "答案")
        db.append_message(session_id, "tool", "observation")

    assert runtime._memory_review_due(session_id) is expected


def test_review_instruction_is_not_frozen_into_snapshot(tmp_path) -> None:
    """回顾提示每 N 轮才出现一次 —— 混进冻结前缀会让前缀漂移（第 7 条的教训）。"""

    db = _db(tmp_path)
    runtime = _runtime(db, tmp_path, 10)
    collection_id = db.upsert_collection("papers")
    session = db.get_session(db.create_session(collection_id=collection_id, mode="deep"))

    snapshot = runtime._prompt_snapshot(session, [collection_id], "deep")
    assert MEMORY_CONTRACT in snapshot
    assert MEMORY_REVIEW_INSTRUCTION not in snapshot

    without = runtime._build_messages(session, [collection_id], "deep", "你好", memory_review=False)
    with_review = runtime._build_messages(session, [collection_id], "deep", "你好", memory_review=True)

    assert MEMORY_REVIEW_INSTRUCTION not in [message["content"] for message in without]
    contents = [message["content"] for message in with_review]
    assert MEMORY_REVIEW_INSTRUCTION in contents
    # 顺序：回顾提示 → 契约提醒 → 当前提问。契约必须是最后一个 system 块，
    # 否则模型会掉出 JSON 信封（第 7 条的实测结论）。
    assert contents[-3] == MEMORY_REVIEW_INSTRUCTION
    assert contents[-2] == REPLY_CONTRACT_REMINDER
    assert with_review[-1]["role"] == "user"


def test_snapshot_version_bumped_for_memory_contract() -> None:
    """旧会话的冻结快照没有记忆规则 —— 不升版本它们永远不会重建。"""

    assert PROMPT_SNAPSHOT_VERSION >= 4


# --------------------------------------------------------------------------- #
# 5) 设置项校验
# --------------------------------------------------------------------------- #

def test_settings_rejects_negative_review_interval(tmp_path) -> None:
    settings = AgentSettings(db_path=str(tmp_path / "agent.db"), memory_review_every_turns=-1)
    with pytest.raises(ValueError):
        settings.validate()


def test_settings_accepts_zero_and_default() -> None:
    AgentSettings(memory_review_every_turns=0).validate()
    assert AgentSettings().memory_review_every_turns == 10
