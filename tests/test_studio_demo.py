from __future__ import annotations

from text2model_forge.studio_demo import create_tutorial_run
from text2model_forge.studio_store import StudioStore


def test_tutorial_run_is_completed_inspectable_and_never_masquerades_as_live_evidence(
    tmp_path,
) -> None:
    store = StudioStore(tmp_path)
    created = create_tutorial_run(store, "tutorial-v1")
    run = store.load(created.run_id)

    assert run.run_mode == "tutorial"
    assert run.state == "completed"
    assert run.stage("D10").state == "approved"
    assert any(item.relative_path.endswith(".glb") for item in run.stage("D10").evidence)
    assert all(
        item.metrics.get("synthetic") is True
        for stage in run.stages
        for item in stage.evidence
    )
    for stage in run.stages:
        for decision in stage.human_decisions:
            assert decision.evidence_hashes == {
                item.evidence_id: item.sha256 for item in stage.evidence
            }


def test_tutorial_run_uses_the_static_prop_contract(tmp_path) -> None:
    run = create_tutorial_run(StudioStore(tmp_path), "tutorial-v2")

    assert run.spec is not None
    assert run.spec.asset_kind == "prop"
    assert run.spec.behavior == "static"
    assert [stage.stage_id for stage in run.stages if not stage.applicable] == [
        "D4",
        "D5",
        "D6",
        "D7",
    ]
