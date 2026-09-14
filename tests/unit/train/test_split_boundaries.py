"""The train and val splits must be exactly complementary for any train_ratio."""

from pathlib import Path
from typing import Literal

import pytest
from datasets import Dataset

from speculators.train.data import ArrowDataset


def _dataset(tmp_path: Path, n: int) -> str:
    ds = Dataset.from_dict(
        {
            "input_ids": [[i, i + 1, i + 2] for i in range(n)],
            "loss_mask": [[1, 1, 1]] * n,
            "seq_len": [3] * n,
        }
    )
    path = tmp_path / "data"
    ds.save_to_disk(str(path))
    (path / "hidden_states").mkdir()
    return str(path)


def _split(path: str, ratio: float, split: Literal["train", "val"]) -> ArrowDataset:
    return ArrowDataset(
        max_len=128,
        datapath=path,
        on_missing="skip",
        train_ratio=ratio,
        split=split,
    )


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
@pytest.mark.parametrize("n", [1000, 12501, 100000])
def test_splits_are_exactly_complementary(tmp_path, ratio, n):
    path = _dataset(tmp_path, n)
    train = _split(path, ratio, "train")
    val = _split(path, ratio, "val")

    assert len(train) + len(val) == n, "splits must partition the dataset"
    assert len(train) == int(n * ratio)
    assert val.start_file_idx == len(train), "val must begin where train ends"
    assert len(train) > 0
    assert len(val) > 0


@pytest.mark.parametrize("ratio", [0.2, 0.9])
def test_no_row_appears_in_both_splits(tmp_path, ratio):
    n = 12501
    path = _dataset(tmp_path, n)
    train = _split(path, ratio, "train")
    val = _split(path, ratio, "val")

    train_ids = {tuple(train.data[i]["input_ids"]) for i in range(len(train))}
    val_ids = {tuple(val.data[i]["input_ids"]) for i in range(len(val))}
    assert not (train_ids & val_ids), "a row is in both train and val"


def test_default_takes_whole_dataset(tmp_path):
    path = _dataset(tmp_path, 100)
    assert len(_split(path, 1.0, "train")) == 100


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_rejects_out_of_range_ratio(tmp_path, ratio):
    path = _dataset(tmp_path, 100)
    with pytest.raises(ValueError, match="train_ratio must be in"):
        _split(path, ratio, "train")


def test_rejects_val_split_with_no_val_data(tmp_path):
    path = _dataset(tmp_path, 100)
    with pytest.raises(ValueError, match="leaves no validation split"):
        _split(path, 1.0, "val")


@pytest.mark.parametrize("ratio", [0.1, 0.5, 0.9])
def test_rejects_empty_train_from_small_dataset(tmp_path, ratio):
    path = _dataset(tmp_path, 1)
    with pytest.raises(ValueError, match="train split is empty"):
        _split(path, ratio, "train")


def test_small_dataset_both_splits_nonempty(tmp_path):
    path = _dataset(tmp_path, 2)
    train = _split(path, 0.5, "train")
    val = _split(path, 0.5, "val")
    assert len(train) == 1
    assert len(val) == 1


def test_explicit_split_column_overrides_index_boundary(tmp_path):
    ds = Dataset.from_dict(
        {
            "input_ids": [[i, i + 1] for i in range(4)],
            "loss_mask": [[1, 1]] * 4,
            "seq_len": [2] * 4,
            "group_id": ["image-a", "image-b", "image-a", "image-c"],
            "data_split": ["train", "val", "train", "val"],
        }
    )
    path = tmp_path / "explicit"
    ds.save_to_disk(str(path))
    (path / "hidden_states").mkdir()

    train = _split(str(path), 0.5, "train")
    val = _split(str(path), 0.5, "val")

    assert train.data["group_id"] == ["image-a", "image-a"]
    assert val.data["group_id"] == ["image-b", "image-c"]
    assert train._map_to_file_idx(0) == 0
    assert train._map_to_file_idx(1) == 2
    assert val._map_to_file_idx(0) == 1
    assert val._map_to_file_idx(1) == 3


def test_explicit_split_rejects_unknown_values(tmp_path):
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3, 4]],
            "loss_mask": [[1, 1], [1, 1]],
            "seq_len": [2, 2],
            "data_split": ["train", "test"],
        }
    )
    path = tmp_path / "invalid-explicit"
    ds.save_to_disk(str(path))

    with pytest.raises(ValueError, match="unsupported values"):
        _split(str(path), 0.9, "train")


def test_explicit_split_rejects_group_leakage(tmp_path):
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3, 4]],
            "loss_mask": [[1, 1], [1, 1]],
            "seq_len": [2, 2],
            "group_id": ["same-image", "same-image"],
            "data_split": ["train", "val"],
        }
    )
    path = tmp_path / "leaked-explicit"
    ds.save_to_disk(str(path))

    with pytest.raises(ValueError, match="appears in both"):
        _split(str(path), 0.9, "train")
