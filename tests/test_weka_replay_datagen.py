# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType
from inference_perf.datagen.weka_trace_replay_datagen import (
    HashIdRandomGenerator,
    WekaTraceReplayDataGenerator,
    WekaTrace,
    RoleSegment,
    ConversationReconstructor,
    longest_common_prefix,
    truncate_synth_buf_at_block,
    _IdleGapTimeWarp,
)


def test_longest_common_prefix() -> None:
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([1, 2], [1, 2, 3]) == 2
    assert longest_common_prefix([1], [2]) == 0
    assert longest_common_prefix([], [1]) == 0


def test_truncate_synth_buf_at_block() -> None:
    def dummy_decode(tokens: list[int]) -> str:
        return ",".join(str(t) for t in tokens)

    segments = [
        RoleSegment(role="system", block_start=0, block_count=2, tokens=[10, 11], content="10,11"),
        RoleSegment(role="user", block_start=2, block_count=3, tokens=[20, 21, 22], content="20,21,22"),
    ]

    # Truncate at block 3
    # block 3 lies inside the second segment (since system has blocks 0,1 and user has 2,3,4)
    # Target blocks = 3, so we keep 3 blocks total: system (2) + user (1)
    # The user segment should be truncated to block_count=1, tokens=[20]
    disturbed = truncate_synth_buf_at_block(segments, target_blocks=3, block_size=1, decode_tokens_to_text=dummy_decode)

    assert disturbed == 1  # Index of disturbed segment
    assert len(segments) == 2
    assert segments[0].block_count == 2
    assert segments[1].block_count == 1
    assert segments[1].tokens == [20]
    assert segments[1].content == "20"


def test_hash_id_random_generator() -> None:
    rng1 = HashIdRandomGenerator(base_seed=42)
    rng1.set_trace_id("trace_xyz")
    rng1.reseed_for_hash_id(1001)
    val1_a = rng1.randrange(1000)
    val1_b = rng1.randrange(1000)

    # Re-instantiate with same seed, trace ID and hash ID
    rng2 = HashIdRandomGenerator(base_seed=42)
    rng2.set_trace_id("trace_xyz")
    rng2.reseed_for_hash_id(1001)
    val2_a = rng2.randrange(1000)
    val2_b = rng2.randrange(1000)

    assert val1_a == val2_a
    assert val1_b == val2_b

    # A different seed, trace ID, or hash ID should produce a different result
    rng3 = HashIdRandomGenerator(base_seed=43)
    rng3.set_trace_id("trace_xyz")
    rng3.reseed_for_hash_id(1001)
    val3_a = rng3.randrange(1000)
    assert val3_a != val1_a


def test_idle_gap_time_warp() -> None:
    # Gap cap = 10s
    warp = _IdleGapTimeWarp(request_starts=[0.0, 5.0, 35.0, 45.0], cap_seconds=10.0)

    # 0.0 maps to 0.0
    assert warp.map(0.0) == 0.0
    # 5.0 maps to 5.0 (since 5.0 - 0.0 = 5.0 <= 10.0)
    assert warp.map(5.0) == 5.0

    # 35.0 has a gap from 5.0 of 30s. Cap is 10s. Excess is 20s.
    # So 35.0 shifts left by 20s to 15.0
    assert warp.map(35.0) == 15.0

    # 45.0 has a gap from 35.0 of 10s (equal to cap). So it shifts left by same 20s to 25.0
    assert warp.map(45.0) == 25.0


def test_conversation_reconstructor() -> None:
    def decode_block_tokens(hash_ids: list[int]) -> list[int]:
        # Simple dummy block decoder: block content is [hash_id]
        return [h for h in hash_ids]

    def sample_partial_tail_tokens(n: int, seed: str) -> list[int]:
        return [99] * n

    def decode_tokens_to_text(tokens: list[int]) -> str:
        return ",".join(str(t) for t in tokens)

    recon = ConversationReconstructor(
        block_size=1,
        decode_block_tokens=decode_block_tokens,
        sample_partial_tail_tokens=sample_partial_tail_tokens,
        decode_tokens_to_text=decode_tokens_to_text,
    )

    # Turn 0: input tokens = 3, hash_ids = [1, 2, 3]
    recon.init_turn_0(
        hash_ids=[1, 2, 3],
        in_tokens=3,
        tool_tokens=0,
        system_tokens=0,
        seed="seed0",
    )

    msgs = recon.snapshot_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "1,2,3"

    # Turn 1: curr_in_tokens = 5, hash_ids = [1, 2, 10, 11, 12], prev_out_tokens = 2
    # LCP of [1, 2, 3] and [1, 2, 10, 11, 12] is [1, 2] (length 2)
    # It will truncate the user segment at block 2, leaving system/user tokens [1, 2]
    # The previous assistant output size is 2. The remaining block capacity is curr_in_tokens - lcp = 3.
    # assistant blocks = min(ceil(2/1), 3) = 2. So assistant blocks = [10, 11].
    # User blocks = remaining = 1. User block = [12].
    recon.advance_turn(
        prev_hash_ids=[1, 2, 3],
        prev_in_tokens=3,
        prev_out_tokens=2,
        curr_hash_ids=[1, 2, 10, 11, 12],
        curr_in_tokens=5,
        seed="seed1",
    )

    msgs = recon.snapshot_messages()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "1,2"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "10,11"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "12"


def test_weka_trace_replay_generator_mock(tmp_path: Path) -> None:
    # Create a mock Weka Trace file
    trace_data = {
        "id": "mock_trace_123",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": 0.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            },
            {
                "t": 1.2,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 8,
                "out": 4,
                "hash_ids": [10, 20, 30, 40],
                "api_time": 0.8,
            },
        ],
    }

    trace_file = tmp_path / "mock_trace.json"
    trace_file.write_text(json.dumps(trace_data))

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [9] * len(x)
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(str(i) for i in x)

    # API config
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)

    # Datagen Config
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    # Setup weka_trace_replay config mock
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
    )
    data_cfg.weka_trace_replay = weka_cfg

    # Initialize generator
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen._get_session(0)
    assert session.source_id == "mock_trace_123"

    # Graph should have 2 events representing the 2 parent turns
    assert len(session.graph.events) == 2
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)

    assert events[0].call.total_input_tokens == 4
    assert events[0].call.expected_output_tokens == 2

    assert events[1].call.total_input_tokens == 8
    assert events[1].call.expected_output_tokens == 4

    # Predecessor edge check
    assert events[1].predecessor_event_ids == [events[0].event_id]


def test_weka_trace_replay_generator_mock_no_warp(tmp_path: Path) -> None:
    # Create a mock Weka Trace file with a huge gap (100 seconds)
    trace_data = {
        "id": "mock_trace_no_warp",
        "models": ["claude-opus-4-8"],
        "block_size": 2,
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": 0.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 4,
                "out": 2,
                "hash_ids": [10, 20],
                "api_time": 0.5,
            },
            {
                "t": 100.1,
                "type": "n",
                "model": "claude-opus-4-8",
                "in": 8,
                "out": 4,
                "hash_ids": [10, 20, 30, 40],
                "api_time": 0.8,
            },
        ],
    }

    trace_file = tmp_path / "mock_trace_no_warp.json"
    trace_file.write_text(json.dumps(trace_data))

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_tokenizer().encode = lambda x: [9] * len(x)
    mock_tokenizer.get_tokenizer().decode = lambda x: "".join(str(i) for i in x)

    # API config
    api_cfg = APIConfig(type=APIType.Chat, streaming=False)

    # Datagen Config with trace_idle_gap_cap_seconds = 0 (disabled)
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    weka_cfg = WekaTraceReplayConfig(
        trace_files=[str(trace_file)],
        default_block_size=2,
        trace_idle_gap_cap_seconds=0,
    )
    data_cfg.weka_trace_replay = weka_cfg

    # Initialize generator
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=mock_tokenizer,
        num_workers=1,
    )

    assert len(gen.sessions) == 1
    session = gen._get_session(0)
    events = sorted(session.graph.events.values(), key=lambda e: e.t_start_ms)

    # In raw seconds: t0 = 0.1s (100ms), t1 = 100.1s (100100ms)
    # The gap is exactly 100000ms. Since gap warping is disabled (<= 0),
    # the start times should reflect original timing.
    assert events[0].t_start_ms == 100
    assert events[1].t_start_ms == 100100
    assert events[1].wait_ms == 100100 - events[0].t_end_ms


def _filter_trace(trace_id: str, peak_in: int, out: int = 2, subagent_in: int = 0) -> Dict[str, Any]:
    requests: List[Dict[str, Any]] = [
        {"t": 0.1, "type": "n", "model": "m", "in": peak_in, "out": out, "hash_ids": [1]},
    ]
    if subagent_in:
        requests.append(
            {
                "t": 0.2,
                "type": "subagent",
                "agent_id": "a1",
                "subagent_type": "Subagent",
                "requests": [{"t": 0.3, "type": "n", "model": "m", "in": subagent_in, "out": out, "hash_ids": [2]}],
            }
        )
    return {"id": trace_id, "models": ["m"], "block_size": 2, "requests": requests}


def _load_with_filter(
    tmp_path: Path, traces: List[Dict[str, Any]], filter_expr: Optional[str], **kwargs: Any
) -> List[WekaTrace]:
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig

    for i, trace in enumerate(traces):
        (tmp_path / f"trace_{i:03d}.json").write_text(json.dumps(trace))

    weka_cfg = WekaTraceReplayConfig(
        trace_directory=str(tmp_path),
        default_block_size=2,
        filter=filter_expr,
        **kwargs,
    )
    gen = WekaTraceReplayDataGenerator.__new__(WekaTraceReplayDataGenerator)
    gen.weka_config = weka_cfg
    return WekaTraceReplayDataGenerator._load_weka_traces(gen)


def test_augment_for_filter_derives_aggregates() -> None:
    blob = _filter_trace("t1", peak_in=100, out=5, subagent_in=300)
    augmented = WekaTraceReplayDataGenerator._augment_for_filter(blob)

    # sub-agent inners count too
    assert augmented["max_tokens"] == 300 + 5
    assert augmented["total_tokens"] == 100 + 300 + 5 + 5
    assert augmented["num_turns"] == 2
    assert augmented["id"] == "t1"


def test_augment_for_filter_does_not_overwrite_recorded_keys() -> None:
    blob = _filter_trace("t1", peak_in=100)
    blob["max_tokens"] = 999
    assert WekaTraceReplayDataGenerator._augment_for_filter(blob)["max_tokens"] == 999


def test_filter_selects_traces_by_max_tokens(tmp_path: Path) -> None:
    traces = [_filter_trace("small", peak_in=100), _filter_trace("big", peak_in=500)]
    kept = _load_with_filter(tmp_path, traces, "lambda x: x['max_tokens'] < 262144")
    assert [t.id for t in kept] == ["small", "big"]

    kept = _load_with_filter(tmp_path, traces, "lambda x: x['max_tokens'] < 200")
    assert [t.id for t in kept] == ["small"]


def test_filter_none_keeps_every_trace(tmp_path: Path) -> None:
    traces = [_filter_trace("a", peak_in=100), _filter_trace("b", peak_in=500)]
    assert len(_load_with_filter(tmp_path, traces, None)) == 2


def test_filter_rejecting_all_traces_yields_nothing(tmp_path: Path) -> None:
    traces = [_filter_trace("a", peak_in=100)]
    assert _load_with_filter(tmp_path, traces, "lambda x: x['max_tokens'] < 1") == []


def test_filter_with_unknown_key_raises(tmp_path: Path) -> None:
    traces = [_filter_trace("a", peak_in=100)]
    with pytest.raises(KeyError, match="benchmark"):
        _load_with_filter(tmp_path, traces, "lambda x: x['benchmark'] == 'swebench'")


def test_malformed_filter_expression_raises(tmp_path: Path) -> None:
    traces = [_filter_trace("a", peak_in=100)]
    with pytest.raises(ValueError, match="Invalid filter expression syntax"):
        _load_with_filter(tmp_path, traces, "lambda x: (")


def test_num_dataset_entries_counts_kept_traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inference_perf.config.datagen.replay import WekaTraceReplayConfig
    from inference_perf.datagen import weka_trace_replay_datagen as mod

    jsonl = tmp_path / "traces.jsonl"
    jsonl.write_text("\n".join(json.dumps(_filter_trace(f"t{i}", peak_in=100 if i % 2 == 0 else 500)) for i in range(10)))
    monkeypatch.setattr(mod, "hf_hub_download", lambda **kwargs: str(jsonl))

    def load(num_entries: int, filter_expr: Optional[str]) -> List[WekaTrace]:
        cfg = WekaTraceReplayConfig(
            hf_dataset_path="org/dataset",
            default_block_size=2,
            num_dataset_entries=num_entries,
            filter=filter_expr,
        )
        gen = WekaTraceReplayDataGenerator.__new__(WekaTraceReplayDataGenerator)
        gen.weka_config = cfg
        return WekaTraceReplayDataGenerator._load_weka_traces(gen)

    assert [t.id for t in load(3, None)] == ["t0", "t1", "t2"]

    # counts accepted traces, scanning deeper to fill the quota
    assert [t.id for t in load(3, "lambda x: x['max_tokens'] < 200")] == ["t0", "t2", "t4"]

    assert len(load(99, "lambda x: x['max_tokens'] < 200")) == 5
