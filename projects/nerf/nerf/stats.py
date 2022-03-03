# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import warnings
from itertools import cycle
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from visdom import Visdom


class AverageMeter:
    """
    Computes and stores the average and current value.
    Tracks the exact history of the added values in every epoch.
    """

    def __init__(self) -> None:
        """
        Initialize the structure with empty history and zero-ed moving average.
        """
        self.history = []
        self.reset()

    def reset(self) -> None:
        """
        Reset the running average meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1, epoch: int = 0) -> None:
        """
        Updates the average meter with a value `val`.

        Args:
            val: A float to be added to the meter.
            n: Represents the number of entities to be added.
            epoch: The epoch to which the number should be added.
        """
        # make sure the history is of the same len as epoch
        while len(self.history) <= epoch:
            self.history.append([])
        self.history[epoch].append(val / n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_epoch_averages(self):
        """
        Returns:
            averages: A list of average values of the metric for each epoch
                in the history buffer.
        """
        if len(self.history) == 0:
            return None
        return [
            (float(np.array(h).mean()) if len(h) > 0 else float("NaN"))
            for h in self.history
        ]


class Stats:
    """
    Stats logging object useful for gathering statistics of training
    a deep network in PyTorch.

    Example:
        ```
        # Init stats structure that logs statistics 'objective' and 'top1e'.
        stats = Stats( ('objective','top1e') )

        network = init_net()  # init a pytorch module (=neural network)
        dataloader = init_dataloader()  # init a dataloader

        for epoch in range(10):

            # start of epoch -> call new_epoch
            stats.new_epoch()

            # Iterate over batches.
            for batch in dataloader:
                # Run a model and save into a dict of output variables "output"
                output = network(batch)

                # stats.update() automatically parses the 'objective' and 'top1e'
                # from the "output" dict and stores this into the db.
                stats.update(output)
                stats.print() # prints the averages over given epoch

            # Stores the training plots into '/tmp/epoch_stats.pdf'
            # and plots into a visdom server running at localhost (if running).
            stats.plot_stats(plot_file='/tmp/epoch_stats.pdf')
        ```
    """

    def __init__(
        self,
        log_vars: List[str],
        verbose: bool = False,
        epoch: int = -1,
        plot_file: Optional[str] = None,
    ) -> None:
        """
        Args:
            log_vars: The list of variable names to be logged.
            verbose: Print status messages.
            epoch: The initial epoch of the object.
            plot_file: The path to the file that will hold the training plots.
        """
        self.verbose = verbose
        self.log_vars = log_vars
        self.plot_file = plot_file
        self.hard_reset(epoch=epoch)

    def reset(self) -> None:
        """
        Called before an epoch to clear current epoch buffers.
        """
        stat_sets = list(self.stats.keys())
        if self.verbose:
            print("stats: epoch %d - reset" % self.epoch)
        self.it = {k: -1 for k in stat_sets}
        for stat_set in stat_sets:
            for stat in self.stats[stat_set]:
                self.stats[stat_set][stat].reset()

        # Set a new timestamp.
        self._epoch_start = time.time()

    def hard_reset(self, epoch: int = -1) -> None:
        """
        Erases all logged data.
        """
        self._epoch_start = None
        self.epoch = epoch
        if self.verbose:
            print("stats: epoch %d - hard reset" % self.epoch)
        self.stats = {}
        self.reset()

    def new_epoch(self) -> None:
        """
        Initializes a new epoch.
        """
        if self.verbose:
            print("stats: new epoch %d" % (self.epoch + 1))
        self.epoch += 1  # increase epoch counter
        self.reset()  # zero the stats

    def _gather_value(self, val):
        if isinstance(val, float):
            pass
        else:
            val = val.data.cpu().numpy()
            val = float(val.sum())
        return val

    def update(self, preds: dict, stat_set: str = "train") -> None:
        """
        Update the internal logs with metrics of a training step.

        Each metric is stored as an instance of an AverageMeter.

        Args:
            preds: Dict of values to be added to the logs.
            stat_set: The set of statistics to be updated (e.g. "train", "val").
        """

        if self.epoch == -1:  # uninitialized
            warnings.warn(
                "self.epoch==-1 means uninitialized stats structure"
                " -> new_epoch() called"
            )
            self.new_epoch()

        if stat_set not in self.stats:
            self.stats[stat_set] = {}
            self.it[stat_set] = -1

        self.it[stat_set] += 1

        epoch = self.epoch
        it = self.it[stat_set]

        for stat in self.log_vars:

            if stat not in self.stats[stat_set]:
                self.stats[stat_set][stat] = AverageMeter()

            if stat == "sec/it":  # compute speed
                elapsed = time.time() - self._epoch_start
                time_per_it = float(elapsed) / float(it + 1)
                val = time_per_it
            else:
                if stat in preds:
                    val = self._gather_value(preds[stat])
                else:
                    val = None

            if val is not None:
                self.stats[stat_set][stat].update(val, epoch=epoch, n=1)

    def print(self, max_it: Optional[int] = None, stat_set: str = "train") -> None:
        """
        Print the current values of all stored stats.

        Args:
            max_it: Maximum iteration number to be displayed.
                If None, the maximum iteration number is not displayed.
            stat_set: The set of statistics to be printed.
        """

        epoch = self.epoch
        stats = self.stats

        str_out = ""

        it = self.it[stat_set]
        stat_str = ""
        stats_print = sorted(stats[stat_set].keys())
        for stat in stats_print:
            if stats[stat_set][stat].count == 0:
                continue
            stat_str += " {0:.12}: {1:1.3f} |".format(stat, stats[stat_set][stat].avg)

        head_str = f"[{stat_set}] | epoch {epoch} | it {it}"
        if max_it:
            head_str += f"/ {max_it}"

        str_out = f"{head_str} | {stat_str}"

        print(str_out)

    def plot_stats(
        self,
        viz: Visdom = None,
        visdom_env: Optional[str] = None,
        plot_file: Optional[str] = None,
    ) -> None:
        """
        Plot the line charts of the history of the stats.

        Args:
            viz: The Visdom object holding the connection to a Visdom server.
            visdom_env: The visdom environment for storing the graphs.
            plot_file: The path to a file with training plots.
        """

        stat_sets = list(self.stats.keys())

        if viz is None:
            withvisdom = False
        elif not viz.check_connection():
            warnings.warn("Cannot connect to the visdom server! Skipping visdom plots.")
            withvisdom = False
        else:
            withvisdom = True

        lines = []

        for stat in self.log_vars:
            vals = []
            stat_sets_now = []
            for stat_set in stat_sets:
                val = self.stats[stat_set][stat].get_epoch_averages()
                if val is None:
                    continue
                else:
                    val = np.array(val).reshape(-1)
                    stat_sets_now.append(stat_set)
                vals.append(val)

            if len(vals) == 0:
                continue

            vals = np.stack(vals, axis=1)
            x = np.arange(vals.shape[0])

            lines.append((stat_sets_now, stat, x, vals))

        if withvisdom:
            for tmodes, stat, x, vals in lines:
                title = "%s" % stat
                opts = {"title": title, "legend": list(tmodes)}
                for i, (tmode, val) in enumerate(zip(tmodes, vals.T)):
                    update = "append" if i > 0 else None
                    valid = np.where(np.isfinite(val))
                    if len(valid) == 0:
                        continue
                    viz.line(
                        Y=val[valid],
                        X=x[valid],
                        env=visdom_env,
                        opts=opts,
                        win=f"stat_plot_{title}",
                        name=tmode,
                        update=update,
                    )

        if plot_file is None:
            plot_file = self.plot_file

        if plot_file is not None:
            print("Exporting stats to %s" % plot_file)
            ncol = 3
            nrow = int(np.ceil(float(len(lines)) / ncol))
            matplotlib.rcParams.update({"font.size": 5})
            color = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
            fig = plt.figure(1)
            plt.clf()
            for idx, (tmodes, stat, x, vals) in enumerate(lines):
                c = next(color)
                plt.subplot(nrow, ncol, idx + 1)
                for vali, vals_ in enumerate(vals.T):
                    c_ = c * (1.0 - float(vali) * 0.3)
                    valid = np.where(np.isfinite(vals_))
                    if len(valid) == 0:
                        continue
                    plt.plot(x[valid], vals_[valid], c=c_, linewidth=1)
                plt.ylabel(stat)
                plt.xlabel("epoch")
                plt.gca().yaxis.label.set_color(c[0:3] * 0.75)
                plt.legend(tmodes)
                gcolor = np.array(mcolors.to_rgba("lightgray"))
                plt.grid(
                    b=True, which="major", color=gcolor, linestyle="-", linewidth=0.4
                )
                plt.grid(
                    b=True, which="minor", color=gcolor, linestyle="--", linewidth=0.2
                )
                plt.minorticks_on()

            plt.tight_layout()
            plt.show()
            fig.savefig(plot_file)
