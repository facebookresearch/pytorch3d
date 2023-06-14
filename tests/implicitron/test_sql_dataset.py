# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import unittest
from collections import Counter

import pkg_resources

import torch

from pytorch3d.implicitron.dataset.sql_dataset import SqlIndexDataset

NO_BLOBS_KWARGS = {
    "dataset_root": "",
    "load_images": False,
    "load_depths": False,
    "load_masks": False,
    "load_depth_masks": False,
    "box_crop": False,
}

logger = logging.getLogger("pytorch3d.implicitron.dataset.sql_dataset")
sh = logging.StreamHandler()
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)


DATASET_ROOT = pkg_resources.resource_filename(__name__, "data/sql_dataset")
METADATA_FILE = os.path.join(DATASET_ROOT, "sql_dataset_100.sqlite")
SET_LIST_FILE = os.path.join(DATASET_ROOT, "set_lists_100.json")


class TestSqlDataset(unittest.TestCase):
    def test_basic(self, sequence="cat1_seq2", frame_number=4):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 100)

        # check the items are consecutive
        past_sequences = set()
        last_frame_number = -1
        last_sequence = ""
        for i in range(len(dataset)):
            item = dataset[i]

            if item.frame_number == 0:
                self.assertNotIn(item.sequence_name, past_sequences)
                past_sequences.add(item.sequence_name)
                last_sequence = item.sequence_name
            else:
                self.assertEqual(item.sequence_name, last_sequence)
                self.assertEqual(item.frame_number, last_frame_number + 1)

            last_frame_number = item.frame_number

        # test indexing
        with self.assertRaises(IndexError):
            dataset[len(dataset) + 1]

        # test sequence-frame indexing
        item = dataset[sequence, frame_number]
        self.assertEqual(item.sequence_name, sequence)
        self.assertEqual(item.frame_number, frame_number)

        with self.assertRaises(IndexError):
            dataset[sequence, 13]

    def test_filter_empty_masks(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 78)

    def test_pick_frames_sql_clause(self):
        dataset_no_empty_masks = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            pick_frames_sql_clause="_mask_mass IS NULL OR _mask_mass > 0",
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        # check the datasets are equal
        self.assertEqual(len(dataset), len(dataset_no_empty_masks))
        for i in range(len(dataset)):
            item_nem = dataset_no_empty_masks[i]
            item = dataset[i]
            self.assertEqual(item_nem.image_path, item.image_path)

        # remove_empty_masks together with the custom criterion
        dataset_ts = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            pick_frames_sql_clause="frame_timestamp < 0.15",
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )
        self.assertEqual(len(dataset_ts), 19)

    def test_limit_categories(self, category="cat0"):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            pick_categories=[category],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 50)
        for i in range(len(dataset)):
            self.assertEqual(dataset[i].sequence_category, category)

    def test_limit_sequences(self, num_sequences=3):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            limit_sequences_to=num_sequences,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 10 * num_sequences)

        def delist(sequence_name):
            return sequence_name if isinstance(sequence_name, str) else sequence_name[0]

        unique_seqs = {delist(dataset[i].sequence_name) for i in range(len(dataset))}
        self.assertEqual(len(unique_seqs), num_sequences)

    def test_pick_exclude_sequencess(self, sequence="cat1_seq2"):
        # pick sequence
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            pick_sequences=[sequence],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 10)
        unique_seqs = {dataset[i].sequence_name for i in range(len(dataset))}
        self.assertCountEqual(unique_seqs, {sequence})

        item = dataset[sequence, 0]
        self.assertEqual(item.sequence_name, sequence)
        self.assertEqual(item.frame_number, 0)

        # exclude sequence
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            exclude_sequences=[sequence],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 90)
        unique_seqs = {dataset[i].sequence_name for i in range(len(dataset))}
        self.assertNotIn(sequence, unique_seqs)

        with self.assertRaises(IndexError):
            dataset[sequence, 0]

    def test_limit_frames(self, num_frames=13):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            limit_to=num_frames,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), num_frames)
        unique_seqs = {dataset[i].sequence_name for i in range(len(dataset))}
        self.assertEqual(len(unique_seqs), 2)

        # test when the limit is not binding
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            limit_to=1000,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 100)

    def test_limit_frames_per_sequence(self, num_frames=2):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            n_frames_per_sequence=num_frames,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), num_frames * 10)
        seq_counts = Counter(dataset[i].sequence_name for i in range(len(dataset)))
        self.assertEqual(len(seq_counts), 10)
        self.assertCountEqual(
            set(seq_counts.values()), {2}
        )  # all counts are num_frames

        with self.assertRaises(IndexError):
            dataset[next(iter(seq_counts)), num_frames + 1]

        # test when the limit is not binding
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            n_frames_per_sequence=13,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )
        self.assertEqual(len(dataset), 100)

    def test_limit_sequence_per_category(self, num_sequences=2):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            limit_sequences_per_category_to=num_sequences,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), num_sequences * 10 * 2)
        seq_names = list(dataset.sequence_names())
        self.assertEqual(len(seq_names), num_sequences * 2)
        # check that we respect the row order
        for seq_name in seq_names:
            self.assertLess(int(seq_name[-1]), num_sequences)

        # test when the limit is not binding
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            limit_sequences_per_category_to=13,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )
        self.assertEqual(len(dataset), 100)

    def test_filter_medley(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            pick_categories=["cat1"],
            exclude_sequences=["cat1_seq0"],  # retaining "cat1_seq1" and on
            limit_sequences_to=2,  # retaining "cat1_seq1" and "cat1_seq2"
            limit_to=14,  # retaining full "cat1_seq1" and 4 from "cat1_seq2"
            n_frames_per_sequence=6,  # cutting "cat1_seq1" to 6 frames
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        # result: preserved 6 frames from cat1_seq1 and 4 from cat1_seq2
        seq_counts = Counter(dataset[i].sequence_name for i in range(len(dataset)))
        self.assertCountEqual(seq_counts.keys(), ["cat1_seq1", "cat1_seq2"])
        self.assertEqual(seq_counts["cat1_seq1"], 6)
        self.assertEqual(seq_counts["cat1_seq2"], 4)

    def test_subsets_trivial(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            subset_lists_file=SET_LIST_FILE,
            limit_to=100,  # force sorting
            subsets=["train", "test"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 100)

        # check the items are consecutive
        past_sequences = set()
        last_frame_number = -1
        last_sequence = ""
        for i in range(len(dataset)):
            item = dataset[i]

            if item.frame_number == 0:
                self.assertNotIn(item.sequence_name, past_sequences)
                past_sequences.add(item.sequence_name)
                last_sequence = item.sequence_name
            else:
                self.assertEqual(item.sequence_name, last_sequence)
                self.assertEqual(item.frame_number, last_frame_number + 1)

            last_frame_number = item.frame_number

    def test_subsets_filter_empty_masks(self):
        # we need to test this case as it uses quite different logic with `df.drop()`
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            subset_lists_file=SET_LIST_FILE,
            subsets=["train", "test"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 78)

    def test_subsets_pick_frames_sql_clause(self):
        dataset_no_empty_masks = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            subset_lists_file=SET_LIST_FILE,
            subsets=["train", "test"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            pick_frames_sql_clause="_mask_mass IS NULL OR _mask_mass > 0",
            subset_lists_file=SET_LIST_FILE,
            subsets=["train", "test"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        # check the datasets are equal
        self.assertEqual(len(dataset), len(dataset_no_empty_masks))
        for i in range(len(dataset)):
            item_nem = dataset_no_empty_masks[i]
            item = dataset[i]
            self.assertEqual(item_nem.image_path, item.image_path)

        # remove_empty_masks together with the custom criterion
        dataset_ts = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            pick_frames_sql_clause="frame_timestamp < 0.15",
            subset_lists_file=SET_LIST_FILE,
            subsets=["train", "test"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset_ts), 19)

    def test_single_subset(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            subset_lists_file=SET_LIST_FILE,
            subsets=["train"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 50)

        with self.assertRaises(IndexError):
            dataset[51]

        # check the items are consecutive
        past_sequences = set()
        last_frame_number = -1
        last_sequence = ""
        for i in range(len(dataset)):
            item = dataset[i]

            if item.frame_number < 2:
                self.assertNotIn(item.sequence_name, past_sequences)
                past_sequences.add(item.sequence_name)
                last_sequence = item.sequence_name
            else:
                self.assertEqual(item.sequence_name, last_sequence)
                self.assertEqual(item.frame_number, last_frame_number + 2)

            last_frame_number = item.frame_number

        item = dataset[last_sequence, 0]
        self.assertEqual(item.sequence_name, last_sequence)

        with self.assertRaises(IndexError):
            dataset[last_sequence, 1]

    def test_subset_with_filters(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            subset_lists_file=SET_LIST_FILE,
            subsets=["train"],
            pick_categories=["cat1"],
            exclude_sequences=["cat1_seq0"],  # retaining "cat1_seq1" and on
            limit_sequences_to=2,  # retaining "cat1_seq1" and "cat1_seq2"
            limit_to=7,  # retaining full train set of "cat1_seq1" and 2 from "cat1_seq2"
            n_frames_per_sequence=3,  # cutting "cat1_seq1" to 3 frames
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        # result: preserved 6 frames from cat1_seq1 and 4 from cat1_seq2
        seq_counts = Counter(dataset[i].sequence_name for i in range(len(dataset)))
        self.assertCountEqual(seq_counts.keys(), ["cat1_seq1", "cat1_seq2"])
        self.assertEqual(seq_counts["cat1_seq1"], 3)
        self.assertEqual(seq_counts["cat1_seq2"], 2)

    def test_visitor(self):
        dataset_sorted = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        sequences = dataset_sorted.sequence_names()
        i = 0
        for seq in sequences:
            last_ts = float("-Inf")
            for ts, _, idx in dataset_sorted.sequence_frames_in_order(seq):
                self.assertEqual(i, idx)
                i += 1
                self.assertGreaterEqual(ts, last_ts)
                last_ts = ts

        # test legacy visitor
        old_indices = None
        for seq in sequences:
            last_ts = float("-Inf")
            rows = dataset_sorted._index.index.get_loc(seq)
            indices = list(range(rows.start or 0, rows.stop, rows.step or 1))
            fn_ts_list = dataset_sorted.get_frame_numbers_and_timestamps(indices)
            self.assertEqual(len(fn_ts_list), len(indices))

            if old_indices:
                # check raising if we ask for multiple sequences
                with self.assertRaises(ValueError):
                    dataset_sorted.get_frame_numbers_and_timestamps(
                        indices + old_indices
                    )

            old_indices = indices

    def test_visitor_subsets(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            limit_to=100,  # force sorting
            subset_lists_file=SET_LIST_FILE,
            subsets=["train", "test"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        sequences = dataset.sequence_names()
        i = 0
        for seq in sequences:
            last_ts = float("-Inf")
            seq_frames = list(dataset.sequence_frames_in_order(seq))
            self.assertEqual(len(seq_frames), 10)
            for ts, _, idx in seq_frames:
                self.assertEqual(i, idx)
                i += 1
                self.assertGreaterEqual(ts, last_ts)
                last_ts = ts

            last_ts = float("-Inf")
            train_frames = list(dataset.sequence_frames_in_order(seq, "train"))
            self.assertEqual(len(train_frames), 5)
            for ts, _, _ in train_frames:
                self.assertGreaterEqual(ts, last_ts)
                last_ts = ts

    def test_category_to_sequence_names(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            subset_lists_file=SET_LIST_FILE,
            subsets=["train", "test"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        cat_to_seqs = dataset.category_to_sequence_names()
        self.assertEqual(len(cat_to_seqs), 2)
        self.assertIn("cat1", cat_to_seqs)
        self.assertEqual(len(cat_to_seqs["cat1"]), 5)

        # check that override preserves the behavior
        cat_to_seqs_base = super(SqlIndexDataset, dataset).category_to_sequence_names()
        self.assertDictEqual(cat_to_seqs, cat_to_seqs_base)

    def test_category_to_sequence_names_filters(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=True,
            subset_lists_file=SET_LIST_FILE,
            exclude_sequences=["cat1_seq0"],
            subsets=["train", "test"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        cat_to_seqs = dataset.category_to_sequence_names()
        self.assertEqual(len(cat_to_seqs), 2)
        self.assertIn("cat1", cat_to_seqs)
        self.assertEqual(len(cat_to_seqs["cat1"]), 4)  # minus one

        # check that override preserves the behavior
        cat_to_seqs_base = super(SqlIndexDataset, dataset).category_to_sequence_names()
        self.assertDictEqual(cat_to_seqs, cat_to_seqs_base)

    def test_meta_access(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            subset_lists_file=SET_LIST_FILE,
            subsets=["train"],
            frame_data_builder_FrameDataBuilder_args=NO_BLOBS_KWARGS,
        )

        self.assertEqual(len(dataset), 50)

        for idx in [10, ("cat0_seq2", 2)]:
            example_meta = dataset.meta[idx]
            example = dataset[idx]
            self.assertEqual(example_meta.sequence_name, example.sequence_name)
            self.assertEqual(example_meta.frame_number, example.frame_number)
            self.assertEqual(example_meta.frame_timestamp, example.frame_timestamp)
            self.assertEqual(example_meta.sequence_category, example.sequence_category)
            torch.testing.assert_close(example_meta.camera.R, example.camera.R)
            torch.testing.assert_close(example_meta.camera.T, example.camera.T)
            torch.testing.assert_close(
                example_meta.camera.focal_length, example.camera.focal_length
            )
            torch.testing.assert_close(
                example_meta.camera.principal_point, example.camera.principal_point
            )

    def test_meta_access_no_blobs(self):
        dataset = SqlIndexDataset(
            sqlite_metadata_file=METADATA_FILE,
            remove_empty_masks=False,
            subset_lists_file=SET_LIST_FILE,
            subsets=["train"],
            frame_data_builder_FrameDataBuilder_args={
                "dataset_root": ".",
                "box_crop": False,  # required by blob-less accessor
            },
        )

        self.assertIsNone(dataset.meta[0].image_rgb)
        self.assertIsNone(dataset.meta[0].fg_probability)
        self.assertIsNone(dataset.meta[0].depth_map)
        self.assertIsNone(dataset.meta[0].sequence_point_cloud)
        self.assertIsNotNone(dataset.meta[0].camera)
