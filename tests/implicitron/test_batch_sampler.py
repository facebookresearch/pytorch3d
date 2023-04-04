# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import numpy as np

import torch
from pytorch3d.implicitron.dataset.data_loader_map_provider import (
    DoublePoolBatchSampler,
)

from pytorch3d.implicitron.dataset.dataset_base import DatasetBase
from pytorch3d.implicitron.dataset.frame_data import FrameData
from pytorch3d.implicitron.dataset.scene_batch_sampler import SceneBatchSampler


@dataclass
class MockFrameAnnotation:
    frame_number: int
    sequence_name: str = "sequence"
    frame_timestamp: float = 0.0


class MockDataset(DatasetBase):
    def __init__(self, num_seq, max_frame_gap=1):
        """
        Makes a gap of max_frame_gap frame numbers in the middle of each sequence
        """
        self.seq_annots = {f"seq_{i}": None for i in range(num_seq)}
        self._seq_to_idx = {
            f"seq_{i}": list(range(i * 10, i * 10 + 10)) for i in range(num_seq)
        }

        # frame numbers within sequence: [0, ..., 4, n, ..., n+4]
        # where n - 4 == max_frame_gap
        frame_nos = list(range(5)) + list(range(4 + max_frame_gap, 9 + max_frame_gap))
        self.frame_annots = [
            {"frame_annotation": MockFrameAnnotation(no)} for no in frame_nos * num_seq
        ]
        for seq_name, idx in self._seq_to_idx.items():
            for i in idx:
                self.frame_annots[i]["frame_annotation"].sequence_name = seq_name

    def get_frame_numbers_and_timestamps(self, idxs, subset_filter=None):
        assert subset_filter is None
        out = []
        for idx in idxs:
            frame_annotation = self.frame_annots[idx]["frame_annotation"]
            out.append(
                (frame_annotation.frame_number, frame_annotation.frame_timestamp)
            )
        return out

    def __getitem__(self, index: int):
        fa = self.frame_annots[index]["frame_annotation"]
        fd = FrameData(
            sequence_name=fa.sequence_name,
            sequence_category="default_category",
            frame_number=torch.LongTensor([fa.frame_number]),
            frame_timestamp=torch.LongTensor([fa.frame_timestamp]),
        )
        return fd


class TestSceneBatchSampler(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dataset_overfit = MockDataset(1)

    def test_overfit(self):
        num_batches = 3
        batch_size = 10
        sampler = SceneBatchSampler(
            self.dataset_overfit,
            batch_size=batch_size,
            num_batches=num_batches,
            images_per_seq_options=[10],  # will try to sample batch_size anyway
        )

        self.assertEqual(len(sampler), num_batches)

        it = iter(sampler)
        for _ in range(num_batches):
            batch = next(it)
            self.assertIsNotNone(batch)
            self.assertEqual(len(batch), batch_size)  # true for our examples
            self.assertTrue(all(idx // 10 == 0 for idx in batch))

        with self.assertRaises(StopIteration):
            batch = next(it)

    def test_multiseq(self):
        for ips_options in [[10], [2], [3], [2, 3, 4]]:
            for sample_consecutive_frames in [True, False]:
                for consecutive_frames_max_gap in [0, 1, 3]:
                    self._test_multiseq_flavour(
                        ips_options,
                        sample_consecutive_frames,
                        consecutive_frames_max_gap,
                    )

    def test_multiseq_gaps(self):
        num_batches = 16
        batch_size = 10
        dataset_multiseq = MockDataset(5, max_frame_gap=3)
        for ips_options in [[10], [2], [3], [2, 3, 4]]:
            debug_info = f" Images per sequence: {ips_options}."

            sampler = SceneBatchSampler(
                dataset_multiseq,
                batch_size=batch_size,
                num_batches=num_batches,
                images_per_seq_options=ips_options,
                sample_consecutive_frames=True,
                consecutive_frames_max_gap=1,
            )

            self.assertEqual(len(sampler), num_batches, msg=debug_info)

            it = iter(sampler)
            for _ in range(num_batches):
                batch = next(it)
                self.assertIsNotNone(batch, "batch is None in" + debug_info)
                if max(ips_options) > 5:
                    # true for our examples
                    self.assertEqual(len(batch), 5, msg=debug_info)
                else:
                    # true for our examples
                    self.assertEqual(len(batch), batch_size, msg=debug_info)

                self._check_frames_are_consecutive(
                    batch, dataset_multiseq.frame_annots, debug_info
                )

    def _test_multiseq_flavour(
        self,
        ips_options,
        sample_consecutive_frames,
        consecutive_frames_max_gap,
        num_batches=16,
        batch_size=10,
    ):
        debug_info = (
            f" Images per sequence: {ips_options}, "
            f"sample_consecutive_frames: {sample_consecutive_frames}, "
            f"consecutive_frames_max_gap: {consecutive_frames_max_gap}, "
        )
        # in this test, either consecutive_frames_max_gap == max_frame_gap,
        # or consecutive_frames_max_gap == 0, so segments consist of full sequences
        frame_gap = consecutive_frames_max_gap if consecutive_frames_max_gap > 0 else 3
        dataset_multiseq = MockDataset(5, max_frame_gap=frame_gap)
        sampler = SceneBatchSampler(
            dataset_multiseq,
            batch_size=batch_size,
            num_batches=num_batches,
            images_per_seq_options=ips_options,
            sample_consecutive_frames=sample_consecutive_frames,
            consecutive_frames_max_gap=consecutive_frames_max_gap,
        )

        self.assertEqual(len(sampler), num_batches, msg=debug_info)

        it = iter(sampler)
        typical_counts = set()
        for _ in range(num_batches):
            batch = next(it)
            self.assertIsNotNone(batch, "batch is None in" + debug_info)
            # true for our examples
            self.assertEqual(len(batch), batch_size, msg=debug_info)
            # find distribution over sequences
            counts = _count_by_quotient(batch, 10)
            freqs = _count_by_quotient(counts.values(), 1)
            self.assertLessEqual(
                len(freqs),
                2,
                msg="We should have maximum of 2 different "
                "frequences of sequences in the batch." + debug_info,
            )
            if len(freqs) == 2:
                most_seq_count = max(*freqs.keys())
                last_seq = min(*freqs.keys())
                self.assertEqual(
                    freqs[last_seq],
                    1,
                    msg="Only one odd sequence allowed." + debug_info,
                )
            else:
                self.assertEqual(len(freqs), 1)
                most_seq_count = next(iter(freqs))

            self.assertIn(most_seq_count, ips_options)
            typical_counts.add(most_seq_count)

            if sample_consecutive_frames:
                self._check_frames_are_consecutive(
                    batch,
                    dataset_multiseq.frame_annots,
                    debug_info,
                    max_gap=consecutive_frames_max_gap,
                )

        self.assertTrue(
            all(i in typical_counts for i in ips_options),
            "Some of the frequency options did not occur among "
            f"the {num_batches} batches (could be just bad luck)." + debug_info,
        )

        with self.assertRaises(StopIteration):
            batch = next(it)

    def _check_frames_are_consecutive(self, batch, annots, debug_info, max_gap=1):
        # make sure that sampled frames are consecutive
        for i in range(len(batch) - 1):
            curr_idx, next_idx = batch[i : i + 2]
            if curr_idx // 10 == next_idx // 10:  # same sequence
                if max_gap > 0:
                    curr_idx, next_idx = [
                        annots[idx]["frame_annotation"].frame_number
                        for idx in (curr_idx, next_idx)
                    ]
                    gap = max_gap
                else:
                    gap = 1  # we'll check that raw dataset indices are consecutive

                self.assertLessEqual(next_idx - curr_idx, gap, msg=debug_info)


def _count_by_quotient(indices, divisor):
    counter = defaultdict(int)
    for i in indices:
        counter[i // divisor] += 1

    return counter


class TestRandomSampling(unittest.TestCase):
    def test_double_pool_batch_sampler(self):
        unknown_idxs = [2, 3, 4, 5, 8]
        known_idxs = [2, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        for replacement, num_batches in product([True, False], [None, 4, 5, 6, 30]):
            with self.subTest(f"{replacement}, {num_batches}"):
                sampler = DoublePoolBatchSampler(
                    first_indices=unknown_idxs,
                    rest_indices=known_idxs,
                    batch_size=4,
                    replacement=replacement,
                    num_batches=num_batches,
                )
                for _ in range(6):
                    epoch = list(sampler)
                    self.assertEqual(len(epoch), num_batches or len(unknown_idxs))
                    for batch in epoch:
                        self.assertEqual(len(batch), 4)
                        self.assertIn(batch[0], unknown_idxs)
                        for i in batch[1:]:
                            self.assertIn(i, known_idxs)
                    if not replacement and 4 != num_batches:
                        self.assertEqual(
                            {batch[0] for batch in epoch}, set(unknown_idxs)
                        )
