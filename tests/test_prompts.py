import numpy as np

from mvp0.prompts import (
    PromptRecord,
    encode_prompts_mock,
    load_prompt_feature_store,
    prompt_records_from_rows,
    write_prompt_feature_store,
    write_prompt_table,
    read_prompt_table,
)


def test_prompt_table_parses_synthetic_rows():
    records = prompt_records_from_rows(
        [
            {
                "task_id": "task_00001",
                "task_meta_text": "Use the gripper to strike the small ball into the tabletop goal.",
                "primitive_1": "adjust",
                "object_1": "tabletop_goal",
                "primitive_2": "contact",
                "object_2": "ball",
            }
        ]
    )

    assert records[0].task_id == "task_00001"
    assert records[0].primitive_chain == ("adjust tabletop_goal", "contact ball")


def test_prompt_format_includes_goal_and_chain_but_not_task_id():
    record = prompt_records_from_rows(
        [
            {
                "task_id": "task_00001",
                "task_meta_text": "Move the cup to the saucer.",
                "primitive_1": "grasp",
                "object_1": "cup",
                "primitive_2": "place",
                "object_2": "saucer",
            }
        ]
    )[0]

    assert "Move the cup to the saucer." in record.prompt
    assert "grasp cup -> place saucer" in record.prompt
    assert "primitive-local DeltaPhi in [0, 1]" in record.prompt
    assert "task_00001" not in record.prompt


def test_prompt_table_round_trip(tmp_path):
    records = [
        PromptRecord(
            task_id="taskA",
            task_meta_text="meta",
            primitive_chain=("grasp cup",),
            prompt="prompt text",
        )
    ]
    path = tmp_path / "prompt_table.jsonl"

    write_prompt_table(records, path)
    loaded = read_prompt_table(path)

    assert loaded == records


def test_mock_prompt_features_are_deterministic(tmp_path):
    records = [
        PromptRecord(task_id="taskA", task_meta_text="meta", primitive_chain=("grasp cup",), prompt="prompt text")
    ]

    first = encode_prompts_mock(records, feature_dim=8, seed=123)
    second = encode_prompts_mock(records, feature_dim=8, seed=123)

    np.testing.assert_allclose(first, second)
    assert first.shape == (1, 8)


def test_prompt_feature_store_round_trip(tmp_path):
    records = [
        PromptRecord(task_id="taskA", task_meta_text="meta", primitive_chain=("grasp cup",), prompt="prompt text"),
        PromptRecord(task_id="taskB", task_meta_text="meta", primitive_chain=("place cup",), prompt="prompt text 2"),
    ]
    features = np.arange(12, dtype=np.float32).reshape(2, 6)
    path = tmp_path / "prompt_features.npz"

    write_prompt_feature_store(path, records, features)
    loaded = load_prompt_feature_store(path, expected_dim=6)

    assert sorted(loaded) == ["taskA", "taskB"]
    np.testing.assert_allclose(loaded["taskB"], features[1])
