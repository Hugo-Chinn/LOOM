import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess = _load_module(
    "taobao_preprocess", "data/Taobao/preprocess_Taobao_seq_action.py")
pairs_module = _load_module(
    "taobao_transition_pairs", "data/Taobao/build_transition_pairs.py")
all_pairs_module = _load_module(
    "taobao_all_transition_pairs", "data/Taobao/build_all_transition_pairs.py")


def test_first_future_event_times_and_temporal_edges():
    exposures = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u3"],
        "cate_id": ["c", "c", "c", "c"],
        "brand": ["b", "b", "b", "b"],
        "time_stamp": [100, 180, 100, 100],
        "is_click": [1.0, 1.0, 1.0, 0.0],
    })
    behavior = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u3"],
        "cate_id": ["c"] * 7,
        "brand": ["b"] * 7,
        "time_stamp": [120, 130, 150, 220, 160, 140, 110],
        "cart": [1, 0, 0, 1, 0, 1, 0],
        "fav": [0, 1, 0, 0, 0, 0, 0],
        "buy": [0, 0, 1, 0, 1, 0, 1],
    })
    result = preprocess.attach_behavior_labels_by_forward_window(
        exposures.copy(), behavior, behavior_label_window=100,
        retain_transition_timestamps=True)

    first = result.iloc[0]
    assert first["first_cart_time"] == 120
    assert first["first_fav_time"] == 130
    assert first["first_buy_time"] == 150
    assert first["time_to_buy"] == 50
    assert first["edge_click_cart"] == 1
    assert first["edge_click_fav"] == 1
    assert first["edge_cart_buy"] == 1
    assert first["edge_fav_buy"] == 1
    assert first["edge_click_buy_direct"] == 0

    second = result.iloc[1]
    assert second["first_cart_time"] == 220
    assert second["time_to_cart"] == 40
    assert second["buy"] == 0

    third = result.iloc[2]
    assert third["first_cart_time"] == 140
    assert third["first_buy_time"] == 160
    assert third["edge_cart_buy"] == 1
    assert third["edge_click_buy_direct"] == 0

    fourth = result.iloc[3]
    assert fourth["buy"] == 1
    assert fourth["edge_click_buy_direct"] == 0


def test_no_behavior_still_emits_zero_edges():
    exposures = pd.DataFrame({
        "user_id": ["u"], "cate_id": ["c"], "brand": ["b"],
        "time_stamp": [100], "is_click": [1.0],
    })
    result = preprocess.attach_behavior_labels_by_forward_window(
        exposures, None, behavior_label_window=100,
        retain_transition_timestamps=True)
    assert result["first_buy_time"].iloc[0] == -1
    assert result["time_to_buy"].iloc[0] == -1
    assert result["edge_click_buy_direct"].iloc[0] == 0


def test_transition_pair_builder_uses_same_stratum_and_child_negative_controls():
    frame = pd.DataFrame({
        "source_row_id": [1, 2, 3, 4, 5],
        "hour": [10, 10, 10, 11, 10],
        "is_click": [1, 1, 1, 1, 1],
        "cart": [1, 0, 0, 0, 1],
        "fav": [0, 0, 0, 0, 0],
        "buy": [1, 0, 0, 0, 0],
        "edge_cart_buy": [1, 0, 0, 0, 0],
        "edge_click_cart": [1, 0, 0, 0, 1],
        "edge_click_fav": [0, 0, 0, 0, 0],
        "edge_fav_buy": [0, 0, 0, 0, 0],
        "edge_click_buy_direct": [0, 0, 0, 0, 0],
    })
    pairs = pairs_module.build_transition_pairs(
        frame, "edge_click_cart", controls_per_positive=2,
        seed=7, strata_columns=("hour",))
    assert set(pairs["positive_source_row_id"]) == {1, 5}
    assert set(pairs["control_source_row_id"]).issubset({2, 3})
    assert (pairs["hour"] == 10).all()


def test_split_block_manager_keeps_transition_sidecar_row_aligned(tmp_path):
    manager = preprocess.SplitBlockManager("train", tmp_path, num_blocks=1)
    split_df = pd.DataFrame({
        "user_index": [0, 0],
        "item_index": [1, 2],
        "seq_len": [0, 1],
        "user_id": [1, 1],
        **{column: [0, 0] for column in preprocess.USER_STATIC_FEATURES},
        **{column: [0, 0] for column in preprocess.CONTEXT_FEATURES},
        "is_click": [1.0, 0.0],
        "cart": [1.0, 0.0],
        "fav": [0.0, 0.0],
        "buy": [0.0, 0.0],
    })
    transition_df = pd.DataFrame({
        "source_row_id": [100, 101],
        "exposure_time": [1000, 1010],
        "is_click": [1.0, 0.0],
        "cart": [1.0, 0.0],
        "fav": [0.0, 0.0],
        "buy": [0.0, 0.0],
        "hour": [10, 10],
        **{column: [-1, -1] for column in preprocess.BEHAVIOR_TIME_COLUMNS},
        **{column: [-1, -1] for column in preprocess.BEHAVIOR_DELAY_COLUMNS},
        **{column: [0, 0] for column in preprocess.TRANSITION_EDGE_COLUMNS},
    })
    user_info = [{
        "user_index": 0,
        "full_item_seq": [1, 2],
        "full_action_seq": [1, 1],
        "full_timestamp_seq": [1000, 1010],
    }]
    manager.append_partition(
        0, split_df, user_info, transition_df=transition_df)
    manager.close_writers()

    data = pd.read_parquet(tmp_path / "train/data/part-00000.parquet")
    sidecar = pd.read_parquet(
        tmp_path / "train/transition_info/part-00000.parquet")
    assert len(data) == len(sidecar) == 2
    assert sidecar["block_id"].tolist() == [0, 0]
    assert sidecar["block_row_index"].tolist() == [0, 1]
    assert sidecar["source_row_id"].tolist() == [100, 101]
    assert "first_buy_time" not in data.columns


def test_build_all_pairs_writes_edge_files_and_audit(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "pairs"
    for split in ("train", "valid", "test"):
        transition_dir = dataset_root / split / "transition_info"
        transition_dir.mkdir(parents=True)
        frame = pd.DataFrame({
            "source_row_id": [1, 2, 3, 4],
            "hour": [10, 10, 10, 11],
            "is_click": [1, 1, 1, 1],
            "cart": [1, 0, 1, 0],
            "fav": [0, 0, 0, 0],
            "buy": [1, 0, 0, 0],
            "time_to_cart": [10, -1, 20, -1],
            "time_to_fav": [-1, -1, -1, -1],
            "time_to_buy": [30, -1, -1, -1],
            "edge_click_cart": [1, 0, 1, 0],
            "edge_click_fav": [0, 0, 0, 0],
            "edge_cart_buy": [1, 0, 0, 0],
            "edge_fav_buy": [0, 0, 0, 0],
            "edge_click_buy_direct": [0, 0, 0, 0],
        })
        frame.to_parquet(transition_dir / "part-00000.parquet", index=False)

    audit = all_pairs_module.build_all_pairs(
        dataset_root,
        output_root,
        controls_per_positive=2,
        seed=7,
        strata_columns=("hour",),
    )
    cart_pairs = pd.read_parquet(
        output_root / "train/edge_click_cart/part-00000.parquet")
    assert len(cart_pairs) == 2
    assert set(cart_pairs["positive_source_row_id"]) == {1, 3}
    assert set(cart_pairs["control_source_row_id"]) == {2}
    assert audit["splits"]["train"]["totals"]["edge_click_cart"][
        "positive_coverage"] == 1.0
    assert audit["splits"]["train"]["blocks"]["part-00000"][
        "delay_seconds"]["time_to_buy"]["p50"] == 30.0
