from __future__ import annotations

import random

from datasets import load_dataset

from benchmark_agents.swebench_vertex.models import SWEbenchInstance


def load_swebench_instances(dataset_name: str, split: str) -> list[SWEbenchInstance]:
    dataset = load_dataset(dataset_name, split=split)
    return [
        SWEbenchInstance.from_dataset_record(
            record=row,
            dataset_name=dataset_name,
            split=split,
        )
        for row in dataset
    ]


def sample_instances(
    instances: list[SWEbenchInstance],
    sample_size: int,
    seed: int,
) -> list[SWEbenchInstance]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size > len(instances):
        raise ValueError(
            f"sample_size={sample_size} is larger than the available dataset size "
            f"({len(instances)})."
        )
    rng = random.Random(seed)
    return rng.sample(instances, sample_size)
