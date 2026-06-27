from pathlib import Path

from agent_runtime.cli import build_parser, main, run_from_args
from agent_runtime.models import AgentRunResult
from dullahan_shared.schemas.query import QueryEnvelope


ROOT = Path(__file__).resolve().parents[3]


def test_cli_run_accepts_limit_overrides() -> None:
    args = build_parser().parse_args(
        [
            "How should recursive execution be inspected?",
            "--repo-root",
            str(ROOT),
            "--max-depth",
            "1",
            "--max-breadth",
            "2",
        ]
    )

    result = run_from_args(args)

    assert len(result.subqueries) == 2
    assert {subquery.depth for subquery in result.subqueries} == {1}
    assert result.trace_id.startswith("trace:")


def test_cli_text_output(capsys) -> None:
    exit_code = main(
        [
            "How should CAL and EDL cooperate?",
            "--repo-root",
            str(ROOT),
            "--max-depth",
            "1",
            "--max-breadth",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Trace: trace:" in captured.out
    assert "Subqueries: 1" in captured.out


def test_cli_json_output(capsys) -> None:
    exit_code = main(
        [
            "Which expert should handle routing?",
            "--repo-root",
            str(ROOT),
            "--max-depth",
            "1",
            "--max-breadth",
            "1",
            "--json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"trace_id": "trace:' in captured.out
    assert '"expert_responses"' in captured.out


def test_cli_remote_transport_uses_http_runtime(monkeypatch) -> None:
    calls = {}

    class FakeRuntime:
        def run(self, request):
            calls["request"] = request
            return AgentRunResult(
                root_query=QueryEnvelope(
                    sender_id="user",
                    query_id="query:root",
                    query=request.query,
                ),
                subqueries=[],
                expert_responses=[],
                trace_id="trace:test",
                spans=[],
                final_response="ok",
            )

    def fake_remote(**kwargs):
        calls["remote_kwargs"] = kwargs
        return FakeRuntime()

    monkeypatch.setattr("agent_runtime.cli.AgentRuntime.remote", fake_remote)

    args = build_parser().parse_args(
        [
            "Remote run?",
            "--repo-root",
            str(ROOT),
            "--transport",
            "http",
            "--cal-url",
            "http://cal.example",
            "--edl-url",
            "http://edl.example",
            "--tool-timeout-seconds",
            "7",
            "--persist-artifacts",
        ]
    )

    result = run_from_args(args)

    assert result.final_response == "ok"
    assert calls["remote_kwargs"]["cal_base_url"] == "http://cal.example"
    assert calls["remote_kwargs"]["edl_base_url"] == "http://edl.example"
    assert calls["remote_kwargs"]["timeout_seconds"] == 7
    assert calls["request"].persist_artifacts
