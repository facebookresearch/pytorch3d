# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import sqlalchemy as sa
import torch
from pytorch3d.implicitron.dataset.dataset_base import DatasetBase

from pytorch3d.implicitron.dataset.frame_data import (  # noqa
    FrameData,
    FrameDataBuilder,
    FrameDataBuilderBase,
)
from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from sqlalchemy.orm import Session

from .orm_types import SqlFrameAnnotation, SqlSequenceAnnotation


logger = logging.getLogger(__name__)


_SET_LISTS_TABLE: str = "set_lists"


@registry.register
class SqlIndexDataset(DatasetBase, ReplaceableBase):  # pyre-ignore
    """
    A dataset with annotations stored as SQLite tables. This is an index-based dataset.
    The length is returned after all sequence and frame filters are applied (see param
    definitions below). Indices can either be ordinal in [0, len), or pairs of
    (sequence_name, frame_number); with the performance of `dataset[i]` and
    `dataset[sequence_name, frame_number]` being same. A faster way to get metadata only
    (without blobs) is `dataset.meta[idx]` indexing; it requires box_crop==False.
    With ordinal indexing, the sequences are NOT guaranteed to span contiguous index
    ranges, and frame numbers are NOT guaranteed to be increasing within a sequence.
    Sequence-aware batch samplers have to use `sequence_[frames|indices]_in_order`
    iterators, which are efficient.

    This functionality requires SQLAlchemy 2.0 or later.

    Metadata-related args:
        sqlite_metadata_file: A SQLite file containing frame and sequence annotation
            tables (mapping to SqlFrameAnnotation and SqlSequenceAnnotation,
            respectively).
        dataset_root: A root directory to look for images, masks, etc. It can be
            alternatively set in `frame_data_builder` args, but this takes precedence.
        subset_lists_file: A JSON/sqlite file containing the lists of frames
            corresponding to different subsets (e.g. train/val/test) of the dataset;
            format: {subset: [(sequence_name, frame_id, file_path)]}. All entries
            must be present in frame_annotation metadata table.
        path_manager: a facade for non-POSIX filesystems.
        subsets: Restrict frames/sequences only to the given list of subsets
            as defined in subset_lists_file (see above). Applied before all other
            filters.
        remove_empty_masks: Removes the frames with no active foreground pixels
            in the segmentation mask (needs frame_annotation.mask.mass to be set;
            null values are retained).
        pick_frames_sql_clause: SQL WHERE clause to constrain frame annotations
            NOTE: This is a potential security risk! The string is passed to the SQL
            engine verbatim. Don’t expose it to end users of your application!
        pick_categories: Restrict the dataset to the given list of categories.
        pick_sequences: A Sequence of sequence names to restrict the dataset to.
        exclude_sequences: A Sequence of the names of the sequences to exclude.
        limit_sequences_per_category_to: Limit the dataset to the first up to N
            sequences within each category (applies after all other sequence filters
            but before `limit_sequences_to`).
        limit_sequences_to: Limit the dataset to the first `limit_sequences_to`
            sequences (after other sequence filters have been applied but before
            frame-based filters).
        limit_to: Limit the dataset to the first #limit_to frames (after other
            filters have been applied, except n_frames_per_sequence).
        n_frames_per_sequence: If > 0, randomly samples `n_frames_per_sequence`
            frames in each sequences uniformly without replacement if it has
            more frames than that; applied after other frame-level filters.
        seed: The seed of the random generator sampling `n_frames_per_sequence`
            random frames per sequence.
    """

    frame_annotations_type: ClassVar[Type[SqlFrameAnnotation]] = SqlFrameAnnotation

    sqlite_metadata_file: str = ""
    dataset_root: Optional[str] = None
    subset_lists_file: str = ""
    eval_batches_file: Optional[str] = None
    path_manager: Any = None
    subsets: Optional[List[str]] = None
    remove_empty_masks: bool = True
    pick_frames_sql_clause: Optional[str] = None
    pick_categories: Tuple[str, ...] = ()

    pick_sequences: Tuple[str, ...] = ()
    exclude_sequences: Tuple[str, ...] = ()
    limit_sequences_per_category_to: int = 0
    limit_sequences_to: int = 0
    limit_to: int = 0
    n_frames_per_sequence: int = -1
    seed: int = 0
    remove_empty_masks_poll_whole_table_threshold: int = 300_000
    # we set it manually in the constructor
    # _index: pd.DataFrame = field(init=False)

    frame_data_builder: FrameDataBuilderBase
    frame_data_builder_class_type: str = "FrameDataBuilder"

    def __post_init__(self) -> None:
        if sa.__version__ < "2.0":
            raise ImportError("This class requires SQL Alchemy 2.0 or later")

        if not self.sqlite_metadata_file:
            raise ValueError("sqlite_metadata_file must be set")

        if self.dataset_root:
            frame_builder_type = self.frame_data_builder_class_type
            getattr(self, f"frame_data_builder_{frame_builder_type}_args")[
                "dataset_root"
            ] = self.dataset_root

        run_auto_creation(self)
        self.frame_data_builder.path_manager = self.path_manager

        # pyre-ignore  # NOTE: sqlite-specific args (read-only mode).
        self._sql_engine = sa.create_engine(
            f"sqlite:///file:{self.sqlite_metadata_file}?mode=ro&uri=true"
        )

        sequences = self._get_filtered_sequences_if_any()

        if self.subsets:
            index = self._build_index_from_subset_lists(sequences)
        else:
            # TODO: if self.subset_lists_file and not self.subsets, it might be faster to
            # still use the concatenated lists, assuming they cover the whole dataset
            index = self._build_index_from_db(sequences)

        if self.n_frames_per_sequence >= 0:
            index = self._stratified_sample_index(index)

        if len(index) == 0:
            raise ValueError(f"There are no frames in the subsets: {self.subsets}!")

        self._index = index.set_index(["sequence_name", "frame_number"])  # pyre-ignore

        self.eval_batches = None  # pyre-ignore
        if self.eval_batches_file:
            self.eval_batches = self._load_filter_eval_batches()

        logger.info(str(self))

    def __len__(self) -> int:
        # pyre-ignore[16]
        return len(self._index)

    def __getitem__(self, frame_idx: Union[int, Tuple[str, int]]) -> FrameData:
        """
        Fetches FrameData by either iloc in the index or by (sequence, frame_no) pair
        """
        return self._get_item(frame_idx, True)

    @property
    def meta(self):
        """
        Allows accessing metadata only without loading blobs using `dataset.meta[idx]`.
        Requires box_crop==False, since in that case, cameras cannot be adjusted
        without loading masks.

        Returns:
            FrameData objects with blob fields like `image_rgb` set to None.

        Raises:
            ValueError if dataset.box_crop is set.
        """
        return SqlIndexDataset._MetadataAccessor(self)

    @dataclass
    class _MetadataAccessor:
        dataset: "SqlIndexDataset"

        def __getitem__(self, frame_idx: Union[int, Tuple[str, int]]) -> FrameData:
            return self.dataset._get_item(frame_idx, False)

    def _get_item(
        self, frame_idx: Union[int, Tuple[str, int]], load_blobs: bool = True
    ) -> FrameData:
        if isinstance(frame_idx, int):
            if frame_idx >= len(self._index):
                raise IndexError(f"index {frame_idx} out of range {len(self._index)}")

            seq, frame = self._index.index[frame_idx]
        else:
            seq, frame, *rest = frame_idx
            if isinstance(frame, torch.LongTensor):
                frame = frame.item()

            if (seq, frame) not in self._index.index:
                raise IndexError(
                    f"Sequence-frame index {frame_idx} not found; was it filtered out?"
                )

            if rest and rest[0] != self._index.loc[(seq, frame), "_image_path"]:
                raise IndexError(f"Non-matching image path in {frame_idx}.")

        stmt = sa.select(self.frame_annotations_type).where(
            self.frame_annotations_type.sequence_name == seq,
            self.frame_annotations_type.frame_number
            == int(frame),  # cast from np.int64
        )
        seq_stmt = sa.select(SqlSequenceAnnotation).where(
            SqlSequenceAnnotation.sequence_name == seq
        )
        with Session(self._sql_engine) as session:
            entry = session.scalars(stmt).one()
            seq_metadata = session.scalars(seq_stmt).one()

        assert entry.image.path == self._index.loc[(seq, frame), "_image_path"]

        frame_data = self.frame_data_builder.build(
            entry, seq_metadata, load_blobs=load_blobs
        )

        # The rest of the fields are optional
        frame_data.frame_type = self._get_frame_type(entry)
        return frame_data

    def __str__(self) -> str:
        # pyre-ignore[16]
        return f"SqlIndexDataset #frames={len(self._index)}"

    def sequence_names(self) -> Iterable[str]:
        """Returns an iterator over sequence names in the dataset."""
        return self._index.index.unique("sequence_name")

    # override
    def category_to_sequence_names(self) -> Dict[str, List[str]]:
        stmt = sa.select(
            SqlSequenceAnnotation.category, SqlSequenceAnnotation.sequence_name
        ).where(  # we limit results to sequences that have frames after all filters
            SqlSequenceAnnotation.sequence_name.in_(self.sequence_names())
        )
        with self._sql_engine.connect() as connection:
            cat_to_seqs = pd.read_sql(stmt, connection)

        return cat_to_seqs.groupby("category")["sequence_name"].apply(list).to_dict()

    # override
    def get_frame_numbers_and_timestamps(
        self, idxs: Sequence[int], subset_filter: Optional[Sequence[str]] = None
    ) -> List[Tuple[int, float]]:
        """
        Implements the DatasetBase method.

        NOTE: Avoid this function as there are more efficient alternatives such as
        querying `dataset[idx]` directly or getting all sequence frames with
        `sequence_[frames|indices]_in_order`.

        Return the index and timestamp in their videos of the frames whose
        indices are given in `idxs`. They need to belong to the same sequence!
        If timestamps are absent, they are replaced with zeros.
        This is used for letting SceneBatchSampler identify consecutive
        frames.

        Args:
            idxs: a sequence int frame index in the dataset (it can be a slice)
            subset_filter: must remain None

        Returns:
            list of tuples of
                - frame index in video
                - timestamp of frame in video, coalesced with 0s

        Raises:
            ValueError if idxs belong to more than one sequence.
        """

        if subset_filter is not None:
            raise NotImplementedError(
                "Subset filters are not supported in SQL Dataset. "
                "We encourage creating a dataset per subset."
            )

        index_slice, _ = self._get_frame_no_coalesced_ts_by_row_indices(idxs)
        # alternatively, we can use `.values.tolist()`, which may be faster
        # but returns a list of lists
        return list(index_slice.itertuples())

    # override
    def sequence_frames_in_order(
        self, seq_name: str, subset_filter: Optional[Sequence[str]] = None
    ) -> Iterator[Tuple[float, int, int]]:
        """
        Overrides the default DatasetBase implementation (we don’t use `_seq_to_idx`).
        Returns an iterator over the frame indices in a given sequence.
        We attempt to first sort by timestamp (if they are available),
        then by frame number.

        Args:
            seq_name: the name of the sequence.
            subset_filter: subset names to filter to

        Returns:
            an iterator over triplets `(timestamp, frame_no, dataset_idx)`,
                where `frame_no` is the index within the sequence, and
                `dataset_idx` is the index within the dataset.
                `None` timestamps are replaced with 0s.
        """
        # TODO: implement sort_timestamp_first? (which would matter if the orders
        # of frame numbers and timestamps are different)
        rows = self._index.index.get_loc(seq_name)
        if isinstance(rows, slice):
            assert rows.stop is not None, "Unexpected result from pandas"
            rows = range(rows.start or 0, rows.stop, rows.step or 1)
        else:
            rows = np.where(rows)[0]

        index_slice, idx = self._get_frame_no_coalesced_ts_by_row_indices(
            rows, seq_name, subset_filter
        )
        index_slice["idx"] = idx

        yield from index_slice.itertuples(index=False)

    # override
    def get_eval_batches(self) -> Optional[List[Any]]:
        """
        This class does not support eval batches with ordinal indices. You can pass
        eval_batches as a batch_sampler to a data_loader since the dataset supports
        `dataset[seq_name, frame_no]` indexing.
        """
        return self.eval_batches

    # override
    def join(self, other_datasets: Iterable[DatasetBase]) -> None:
        raise ValueError("Not supported! Preprocess the data by merging them instead.")

    # override
    @property
    def frame_data_type(self) -> Type[FrameData]:
        return self.frame_data_builder.frame_data_type

    def is_filtered(self) -> bool:
        """
        Returns `True` in case the dataset has been filtered and thus some frame
        annotations stored on the disk might be missing in the dataset object.
        Does not account for subsets.

        Returns:
            is_filtered: `True` if the dataset has been filtered, else `False`.
        """
        return (
            self.remove_empty_masks
            or self.limit_to > 0
            or self.limit_sequences_to > 0
            or self.limit_sequences_per_category_to > 0
            or len(self.pick_sequences) > 0
            or len(self.exclude_sequences) > 0
            or len(self.pick_categories) > 0
            or self.n_frames_per_sequence > 0
        )

    def _get_filtered_sequences_if_any(self) -> Optional[pd.Series]:
        # maximum possible filter (if limit_sequences_per_category_to == 0):
        # WHERE category IN 'self.pick_categories'
        # AND sequence_name IN 'self.pick_sequences'
        # AND sequence_name NOT IN 'self.exclude_sequences'
        # LIMIT 'self.limit_sequence_to'

        where_conditions = [
            *self._get_category_filters(),
            *self._get_pick_filters(),
            *self._get_exclude_filters(),
        ]

        def add_where(stmt):
            return stmt.where(*where_conditions) if where_conditions else stmt

        if self.limit_sequences_per_category_to <= 0:
            stmt = add_where(sa.select(SqlSequenceAnnotation.sequence_name))
        else:
            subquery = sa.select(
                SqlSequenceAnnotation.sequence_name,
                sa.func.row_number()
                .over(
                    order_by=sa.text("ROWID"),  # NOTE: ROWID is SQLite-specific
                    partition_by=SqlSequenceAnnotation.category,
                )
                .label("row_number"),
            )

            subquery = add_where(subquery).subquery()
            stmt = sa.select(subquery.c.sequence_name).where(
                subquery.c.row_number <= self.limit_sequences_per_category_to
            )

        if self.limit_sequences_to > 0:
            logger.info(
                f"Limiting dataset to first {self.limit_sequences_to} sequences"
            )
            # NOTE: ROWID is SQLite-specific
            stmt = stmt.order_by(sa.text("ROWID")).limit(self.limit_sequences_to)

        if (
            not where_conditions
            and self.limit_sequences_to <= 0
            and self.limit_sequences_per_category_to <= 0
        ):
            # we will not need to filter by sequences
            return None

        with self._sql_engine.connect() as connection:
            sequences = pd.read_sql_query(stmt, connection)["sequence_name"]
        logger.info("... retained %d sequences" % len(sequences))

        return sequences

    def _get_category_filters(self) -> List[sa.ColumnElement]:
        if not self.pick_categories:
            return []

        logger.info(f"Limiting dataset to categories: {self.pick_categories}")
        return [SqlSequenceAnnotation.category.in_(self.pick_categories)]

    def _get_pick_filters(self) -> List[sa.ColumnElement]:
        if not self.pick_sequences:
            return []

        logger.info(f"Limiting dataset to sequences: {self.pick_sequences}")
        return [SqlSequenceAnnotation.sequence_name.in_(self.pick_sequences)]

    def _get_exclude_filters(self) -> List[sa.ColumnOperators]:
        if not self.exclude_sequences:
            return []

        logger.info(f"Removing sequences from the dataset: {self.exclude_sequences}")
        return [SqlSequenceAnnotation.sequence_name.notin_(self.exclude_sequences)]

    def _load_subsets_from_json(self, subset_lists_path: str) -> pd.DataFrame:
        assert self.subsets is not None
        with open(subset_lists_path, "r") as f:
            subset_to_seq_frame = json.load(f)

        seq_frame_list = sum(
            (
                [(*row, subset) for row in subset_to_seq_frame[subset]]
                for subset in self.subsets
            ),
            [],
        )
        index = pd.DataFrame(
            seq_frame_list,
            columns=["sequence_name", "frame_number", "_image_path", "subset"],
        )
        return index

    def _load_subsets_from_sql(self, subset_lists_path: str) -> pd.DataFrame:
        subsets = self.subsets
        assert subsets is not None
        # we need a new engine since we store the subsets in a separate DB
        engine = sa.create_engine(f"sqlite:///{subset_lists_path}")
        table = sa.Table(_SET_LISTS_TABLE, sa.MetaData(), autoload_with=engine)
        stmt = sa.select(table).where(table.c.subset.in_(subsets))
        with engine.connect() as connection:
            index = pd.read_sql(stmt, connection)

        return index

    def _build_index_from_subset_lists(
        self, sequences: Optional[pd.Series]
    ) -> pd.DataFrame:
        if not self.subset_lists_file:
            raise ValueError("Requested subsets but subset_lists_file not given")

        logger.info(f"Loading subset lists from {self.subset_lists_file}.")

        subset_lists_path = self._local_path(self.subset_lists_file)
        if subset_lists_path.lower().endswith(".json"):
            index = self._load_subsets_from_json(subset_lists_path)
        else:
            index = self._load_subsets_from_sql(subset_lists_path)
        index = index.set_index(["sequence_name", "frame_number"])
        logger.info(f"  -> loaded {len(index)} samples of {self.subsets}.")

        if sequences is not None:
            logger.info("Applying filtered sequences.")
            sequence_values = index.index.get_level_values("sequence_name")
            index = index.loc[sequence_values.isin(sequences)]
            logger.info(f"  -> retained {len(index)} samples.")

        pick_frames_criteria = []
        if self.remove_empty_masks:
            logger.info("Culling samples with empty masks.")

            if len(index) > self.remove_empty_masks_poll_whole_table_threshold:
                # APPROACH 1: find empty masks and drop indices.
                # dev load: 17s / 15 s (3.1M / 500K)
                stmt = sa.select(
                    self.frame_annotations_type.sequence_name,
                    self.frame_annotations_type.frame_number,
                ).where(self.frame_annotations_type._mask_mass == 0)
                with Session(self._sql_engine) as session:
                    to_remove = session.execute(stmt).all()

                # Pandas uses np.int64 for integer types, so we have to case
                # we might want to read it to pandas DataFrame directly to avoid the loop
                to_remove = [(seq, np.int64(fr)) for seq, fr in to_remove]
                index.drop(to_remove, errors="ignore", inplace=True)
            else:
                # APPROACH 3: load index into a temp table and join with annotations
                # dev load: 94 s / 23 s (3.1M / 500K)
                pick_frames_criteria.append(
                    sa.or_(
                        self.frame_annotations_type._mask_mass.is_(None),
                        self.frame_annotations_type._mask_mass != 0,
                    )
                )

        if self.pick_frames_sql_clause:
            logger.info("Applying the custom SQL clause.")
            pick_frames_criteria.append(sa.text(self.pick_frames_sql_clause))

        if pick_frames_criteria:
            index = self._pick_frames_by_criteria(index, pick_frames_criteria)

        logger.info(f"  -> retained {len(index)} samples.")

        if self.limit_to > 0:
            logger.info(f"Limiting dataset to first {self.limit_to} frames")
            index = index.sort_index().iloc[: self.limit_to]

        return index.reset_index()

    def _pick_frames_by_criteria(self, index: pd.DataFrame, criteria) -> pd.DataFrame:
        IndexTable = self._get_temp_index_table_instance()
        with self._sql_engine.connect() as connection:
            IndexTable.create(connection)
            # we don’t let pandas’s `to_sql` create the table automatically as
            # the table would be permanent, so we create it and append with pandas
            n_rows = index.to_sql(IndexTable.name, connection, if_exists="append")
            assert n_rows == len(index)
            sa_type = self.frame_annotations_type
            stmt = (
                sa.select(IndexTable)
                .select_from(
                    IndexTable.join(
                        self.frame_annotations_type,
                        sa.and_(
                            sa_type.sequence_name == IndexTable.c.sequence_name,
                            sa_type.frame_number == IndexTable.c.frame_number,
                        ),
                    )
                )
                .where(*criteria)
            )
            return pd.read_sql_query(stmt, connection).set_index(
                ["sequence_name", "frame_number"]
            )

    def _build_index_from_db(self, sequences: Optional[pd.Series]):
        logger.info("Loading sequcence-frame index from the database")
        stmt = sa.select(
            self.frame_annotations_type.sequence_name,
            self.frame_annotations_type.frame_number,
            self.frame_annotations_type._image_path,
            sa.null().label("subset"),
        )
        where_conditions = []
        if sequences is not None:
            logger.info("  applying filtered sequences")
            where_conditions.append(
                self.frame_annotations_type.sequence_name.in_(sequences.tolist())
            )

        if self.remove_empty_masks:
            logger.info("  excluding samples with empty masks")
            where_conditions.append(
                sa.or_(
                    self.frame_annotations_type._mask_mass.is_(None),
                    self.frame_annotations_type._mask_mass != 0,
                )
            )

        if self.pick_frames_sql_clause:
            logger.info("  applying custom SQL clause")
            where_conditions.append(sa.text(self.pick_frames_sql_clause))

        if where_conditions:
            stmt = stmt.where(*where_conditions)

        if self.limit_to > 0:
            logger.info(f"Limiting dataset to first {self.limit_to} frames")
            stmt = stmt.order_by(
                self.frame_annotations_type.sequence_name,
                self.frame_annotations_type.frame_number,
            ).limit(self.limit_to)

        with self._sql_engine.connect() as connection:
            index = pd.read_sql_query(stmt, connection)

        logger.info(f"  -> loaded {len(index)} samples.")
        return index

    def _sort_index_(self, index):
        logger.info("Sorting the index by sequence and frame number.")
        index.sort_values(["sequence_name", "frame_number"], inplace=True)
        logger.info("  -> Done.")

    def _load_filter_eval_batches(self):
        assert self.eval_batches_file
        logger.info(f"Loading eval batches from {self.eval_batches_file}")

        if not os.path.isfile(self.eval_batches_file):
            # The batch indices file does not exist.
            # Most probably the user has not specified the root folder.
            raise ValueError(
                f"Looking for dataset json file in {self.eval_batches_file}. "
                + "Please specify a correct dataset_root folder."
            )

        with open(self.eval_batches_file, "r") as f:
            eval_batches = json.load(f)

        # limit the dataset to sequences to allow multiple evaluations in one file
        pick_sequences = set(self.pick_sequences)
        if self.pick_categories:
            cat_to_seq = self.category_to_sequence_names()
            pick_sequences.update(
                seq for cat in self.pick_categories for seq in cat_to_seq[cat]
            )

        if pick_sequences:
            old_len = len(eval_batches)
            eval_batches = [b for b in eval_batches if b[0][0] in pick_sequences]
            logger.warn(
                f"Picked eval batches by sequence/cat: {old_len} -> {len(eval_batches)}"
            )

        if self.exclude_sequences:
            old_len = len(eval_batches)
            exclude_sequences = set(self.exclude_sequences)
            eval_batches = [b for b in eval_batches if b[0][0] not in exclude_sequences]
            logger.warn(
                f"Excluded eval batches by sequence: {old_len} -> {len(eval_batches)}"
            )

        return eval_batches

    def _stratified_sample_index(self, index):
        # NOTE this stratified sampling can be done more efficiently in
        # the no-subset case above if it is added to the SQL query.
        # We keep this generic implementation since no-subset case is uncommon
        index = index.groupby("sequence_name", group_keys=False).apply(
            lambda seq_frames: seq_frames.sample(
                min(len(seq_frames), self.n_frames_per_sequence),
                random_state=(
                    _seq_name_to_seed(seq_frames.iloc[0]["sequence_name"]) + self.seed
                ),
            )
        )
        logger.info(f"  -> retained {len(index)} samples aster stratified sampling.")
        return index

    def _get_frame_type(self, entry: SqlFrameAnnotation) -> Optional[str]:
        return self._index.loc[(entry.sequence_name, entry.frame_number), "subset"]

    def _get_frame_no_coalesced_ts_by_row_indices(
        self,
        idxs: Sequence[int],
        seq_name: Optional[str] = None,
        subset_filter: Union[Sequence[str], str, None] = None,
    ) -> Tuple[pd.DataFrame, Sequence[int]]:
        """
        Loads timestamps for given index rows belonging to the same sequence.
        If seq_name is known, it speeds up the computation.
        Raises ValueError if `idxs` do not all belong to a single sequences .
        """
        index_slice = self._index.iloc[idxs]
        if subset_filter is not None:
            if isinstance(subset_filter, str):
                subset_filter = [subset_filter]
            indicator = index_slice["subset"].isin(subset_filter)
            index_slice = index_slice.loc[indicator]
            idxs = [i for i, isin in zip(idxs, indicator) if isin]

        frames = index_slice.index.get_level_values("frame_number").tolist()
        if seq_name is None:
            seq_name_list = index_slice.index.get_level_values("sequence_name").tolist()
            seq_name_set = set(seq_name_list)
            if len(seq_name_set) > 1:
                raise ValueError("Given indices belong to more than one sequence.")
            elif len(seq_name_set) == 1:
                seq_name = seq_name_list[0]

        coalesced_ts = sa.sql.functions.coalesce(
            self.frame_annotations_type.frame_timestamp, 0
        )
        stmt = sa.select(
            coalesced_ts.label("frame_timestamp"),
            self.frame_annotations_type.frame_number,
        ).where(
            self.frame_annotations_type.sequence_name == seq_name,
            self.frame_annotations_type.frame_number.in_(frames),
        )

        with self._sql_engine.connect() as connection:
            frame_no_ts = pd.read_sql_query(stmt, connection)

        if len(frame_no_ts) != len(index_slice):
            raise ValueError(
                "Not all indices are found in the database; "
                "do they belong to more than one sequence?"
            )

        return frame_no_ts, idxs

    def _local_path(self, path: str) -> str:
        if self.path_manager is None:
            return path
        return self.path_manager.get_local_path(path)

    def _get_temp_index_table_instance(self, table_name: str = "__index"):
        CachedTable = self.frame_annotations_type.metadata.tables.get(table_name)
        if CachedTable is not None:  # table definition is not idempotent
            return CachedTable

        return sa.Table(
            table_name,
            self.frame_annotations_type.metadata,
            sa.Column("sequence_name", sa.String, primary_key=True),
            sa.Column("frame_number", sa.Integer, primary_key=True),
            sa.Column("_image_path", sa.String),
            sa.Column("subset", sa.String),
            prefixes=["TEMP"],  # NOTE SQLite specific!
        )


def _seq_name_to_seed(seq_name) -> int:
    """Generates numbers in [0, 2 ** 28)"""
    return int(hashlib.sha1(seq_name.encode("utf-8")).hexdigest()[:7], 16)


def _safe_as_tensor(data, dtype):
    return torch.tensor(data, dtype=dtype) if data is not None else None
