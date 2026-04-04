import tempfile
from pathlib import Path


def test_department_prompt_contract_includes_dynamic_operating_sections():
    from src.agents.prompts.department_contracts import (
        compose_department_head_prompt,
        get_department_prompt_seed,
    )

    prompt = compose_department_head_prompt(
        department="research",
        base_prompt=get_department_prompt_seed(
            department="research",
            persona_name="The Innovation Pioneer",
            tagline="Tomorrow's alpha is discovered today",
            communication_style="Excited and forward-thinking",
        ),
        sub_agents=["strategy_researcher", "market_analyst", "backtester"],
    )

    assert "interactive_session" in prompt
    assert "workflow_session" in prompt
    assert "manifest-first" in prompt
    assert "search workspace resources naturally" in prompt
    assert "write OPINION nodes" in prompt
    assert "Department Mail" in prompt
    assert "Kanban" in prompt
    assert "compaction" in prompt.lower()
    assert "financial_data_fetch" in prompt
    assert "context7" in prompt


def test_floor_manager_prompt_contract_uses_direct_access_topology():
    from src.agents.prompts.department_contracts import compose_floor_manager_prompt

    prompt = compose_floor_manager_prompt()

    assert "User access model" in prompt
    assert "The user may talk directly to Copilot, Floor Manager, Department Heads, or Sub-agents" in prompt
    assert "cross-department router" in prompt
    assert "Department Mail" in prompt
    assert "Skill Forge" in prompt


def test_subagent_prompt_contract_is_bounded_and_manifest_first():
    from src.agents.prompts.department_contracts import compose_subagent_prompt

    prompt = compose_subagent_prompt(
        agent_type="strategy_researcher",
        department="research",
    )

    assert "bounded worker sub-agent" in prompt
    assert "Escalate back to your Department Head" in prompt
    assert "Do not request or assume the full canvas payload" in prompt
    assert "resource ids, paths, and manifest entries" in prompt


def test_department_head_build_system_prompt_uses_dynamic_contract(monkeypatch):
    from src.agents.departments.heads.base import DepartmentHead
    from src.agents.departments.types import Department, DepartmentHeadConfig

    monkeypatch.setattr(
        "src.api.settings_endpoints.load_settings",
        lambda: {"agents": {"system_prompts": {}}},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        head = DepartmentHead(
            config=DepartmentHeadConfig(
                department=Department.RESEARCH,
                agent_type="research_head",
                system_prompt="You are the Research Department Head.",
                sub_agents=["strategy_researcher"],
            ),
            mail_db_path=str(Path(tmpdir) / "mail.db"),
        )

        prompt = head._build_system_prompt(
            canvas_context={"canvas": "research", "session_type": "interactive_session"},
            memory_nodes=[{"content": "Remember the London open drift pattern."}],
        )

        assert "interactive_session" in prompt
        assert "workflow_session" in prompt
        assert "Remember the London open drift pattern." in prompt
        assert "Current Canvas Context (manifest-first summary)" in prompt
        assert "Department Mail" in prompt

        head.close()
