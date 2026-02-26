from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedLanguageBalancedSampler(Sampler[int]):
    """DDP-aware sampler that balances languages at sample time.

    Sampling process per rank:
      1) sample language uniformly from local non-empty language buckets
      2) sample row uniformly within that language via shuffled cyclic traversal

    Language buckets are first sharded across ranks (round-robin per language),
    guaranteeing no cross-rank index overlap.
    """

    def __init__(
        self,
        dataset,
        *,
        language_column: str = "lang",
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank}, expected in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.language_column = language_column
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if not hasattr(dataset, "frame"):
            raise ValueError("DistributedLanguageBalancedSampler requires dataset.frame")
        if language_column not in dataset.frame.columns:
            raise ValueError(f"Dataset frame must include '{language_column}' column")

        self._global_lang_to_indices: dict[str, list[int]] = defaultdict(list)
        langs = dataset.frame[language_column].tolist()
        for idx, lang in enumerate(langs):
            self._global_lang_to_indices[str(lang)].append(idx)

        self.languages = sorted(self._global_lang_to_indices.keys())
        self._local_lang_to_indices: dict[str, list[int]] = {}
        for lang in self.languages:
            global_indices = self._global_lang_to_indices[lang]
            self._local_lang_to_indices[lang] = global_indices[self.rank::self.num_replicas]

        self._local_languages = [lang for lang in self.languages if self._local_lang_to_indices[lang]]

        dataset_len = len(self.dataset)
        if self.drop_last:
            self.num_samples = dataset_len // self.num_replicas
        else:
            self.num_samples = (dataset_len + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

        if self.num_samples > 0 and len(self._local_languages) == 0:
            raise RuntimeError(
                f"Rank {self.rank} has no local samples after language sharding. "
                "Reduce world size or adjust dataset composition."
            )

    def __iter__(self) -> Iterator[int]:
        if self.num_samples == 0:
            return iter([])

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        local_buckets: dict[str, list[int]] = {}
        bucket_pos: dict[str, int] = {}
        for lang in self._local_languages:
            values = list(self._local_lang_to_indices[lang])
            if self.shuffle and values:
                perm = torch.randperm(len(values), generator=generator).tolist()
                values = [values[i] for i in perm]
            local_buckets[lang] = values
            bucket_pos[lang] = 0

        sampled: list[int] = []
        lang_tensor_size = len(self._local_languages)

        while len(sampled) < self.num_samples:
            lang_idx = int(torch.randint(0, lang_tensor_size, (1,), generator=generator).item())
            lang = self._local_languages[lang_idx]
            bucket = local_buckets[lang]

            if not bucket:
                continue

            pos = bucket_pos[lang]
            if pos >= len(bucket):
                # Bucket exhausted for this rank: reshuffle and cycle.
                if self.shuffle and len(bucket) > 1:
                    perm = torch.randperm(len(bucket), generator=generator).tolist()
                    bucket = [bucket[i] for i in perm]
                    local_buckets[lang] = bucket
                pos = 0
                bucket_pos[lang] = 0

            sampled.append(bucket[pos])
            bucket_pos[lang] = pos + 1

        return iter(sampled)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
