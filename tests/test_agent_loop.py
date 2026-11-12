"""Tests the agent loop end to end with a scripted stand-in for Claude (no API key, no cost)."""

from types import SimpleNamespace

import stadium_pass.agent as agent


def _tool_use(id_, name, inp):
    """Build a fake tool_use block, shaped like the API's.

    Args:
        id_: Tool call id.
        name: Tool name.
        inp: Tool inputs.

    Returns:
        An object with the same fields the agent reads from a real block.
    """
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def _text(t):
    """Build a fake text block.

    Args:
        t: The text.

    Returns:
        An object with the same fields the agent reads from a real block.
    """
    return SimpleNamespace(type="text", text=t)


class _ScriptedClaude:
    """Stands in for anthropic.Anthropic and replays a fixed conversation."""

    def __init__(self, *args, **kwargs):
        """Load the script: list sections, plan a seat with a typo, fix it, write files, finish."""
        self.turns = [
            ([_tool_use("1", "list_sections", {})], "tool_use"),
            ([_tool_use("2", "plan_seat", {"name": "sec112_rowF", "section": "1120", "row": "F"})], "tool_use"),
            ([_tool_use("3", "plan_seat", {"name": "sec112_rowF", "section": "112", "row": "F",
                                           "look_at": "near_hoop"})], "tool_use"),
            ([_tool_use("4", "write_camera_files", {})], "tool_use"),
            ([_text("Section 112, Row F: behind the east baseline, six rows back.")], "end_turn"),
        ]
        self.seen_errors = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        """Return the next scripted reply, and note any tool errors sent back to "Claude".

        Args:
            **kwargs: The Messages API arguments the agent sent.

        Returns:
            An object shaped like an API response.
        """
        last = kwargs["messages"][-1]["content"]
        if isinstance(last, list):
            self.seen_errors += [b for b in last if isinstance(b, dict) and b.get("is_error")]
        content, stop = self.turns.pop(0)
        return SimpleNamespace(content=content, stop_reason=stop)


def test_agent_recovers_from_bad_section_and_writes_files(tmp_path, monkeypatch):
    """A bad section is reported back as an error, the retry succeeds, and camera files are written."""
    fake = _ScriptedClaude()
    monkeypatch.setattr(agent.anthropic, "Anthropic", lambda *a, **k: fake)
    run_dir = agent.run_agent(["Section 112, Row F"], out_root=str(tmp_path))

    assert len(fake.seen_errors) == 1 and "Unknown section" in fake.seen_errors[0]["content"]
    assert (run_dir / "cameras" / "sec112_rowF" / "images.txt").exists()
    assert (run_dir / "log.jsonl").read_text().count("tool_call") == 4
