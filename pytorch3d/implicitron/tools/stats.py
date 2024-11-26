# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import gzip
import json
import logging
import time
import warnings
from collections.abc import Iterable
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from pytorch3d.implicitron.tools.vis_utils import get_visdom_connection

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.history = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, epoch=0):
        # make sure the history is of the same len as epoch
        while len(self.history) <= epoch:
            self.history.append([])

        self.history[epoch].append(val / n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_epoch_averages(self, epoch=-1):
        if len(self.history) == 0:  # no stats here
            return None
        elif epoch == -1:
            return [
                (float(np.array(x).mean()) if len(x) > 0 else float("NaN"))
                for x in self.history
            ]
        else:
            return float(np.array(self.history[epoch]).mean())

    def get_all_values(self):
        all_vals = [np.array(x) for x in self.history]
        all_vals = np.concatenate(all_vals)
        return all_vals

    def get_epoch(self):
        return len(self.history)

    @staticmethod
    def from_json_str(json_str):
        self = AverageMeter()
        self.__dict__.update(json.loads(json_str))
        return self


class Stats:
    # TODO: update this with context manager
    """
    stats logging object useful for gathering statistics of training a deep net in pytorch
    Example::

        # init stats structure that logs statistics 'objective' and 'top1e'
        stats = Stats( ('objective','top1e') )
        network = init_net() # init a pytorch module (=nueral network)
        dataloader = init_dataloader() # init a dataloader
        for epoch in range(10):
            # start of epoch -> call new_epoch
            stats.new_epoch()

            # iterate over batches
            for batch in dataloader:

                output = network(batch) # run and save into a dict of output variables

                # stats.update() automatically parses the 'objective' and 'top1e' from
                # the "output" dict and stores this into the db
                stats.update(output)
                # prints the metric averages over given epoch
                std_out = stats.get_status_string()
                logger.info(str_out)
            # stores the training plots into '/tmp/epoch_stats.pdf'
            # and plots into a visdom server running at localhost (if running)
            stats.plot_stats(plot_file='/tmp/epoch_stats.pdf')

    """

    def __init__(
        self,
        log_vars,
        epoch=-1,
        visdom_env="main",
        do_plot=True,
        plot_file=None,
        visdom_server="http://localhost",
        visdom_port=8097,
    ):
        self.log_vars = log_vars
        self.visdom_env = visdom_env
        self.visdom_server = visdom_server
        self.visdom_port = visdom_port
        self.plot_file = plot_file
        self.do_plot = do_plot
        self.hard_reset(epoch=epoch)
        self._t_last_update = None

    @staticmethod
    def from_json_str(json_str):
        self = Stats([])
        # load the global state
        self.__dict__.update(json.loads(json_str))
        # recover the AverageMeters
        for stat_set in self.stats:
            self.stats[stat_set] = {
                log_var: AverageMeter.from_json_str(log_vals_json_str)
                for log_var, log_vals_json_str in self.stats[stat_set].items()
            }
        return self

    @staticmethod
    def load(flpath, postfix=".jgz"):
        flpath = _get_postfixed_filename(flpath, postfix)
        with gzip.open(flpath, "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))
        return Stats.from_json_str(data)

    def save(self, flpath, postfix=".jgz"):
        flpath = _get_postfixed_filename(flpath, postfix)
        # store into a gzipped-json
        with gzip.open(flpath, "w") as fout:
            fout.write(json.dumps(self, cls=StatsJSONEncoder).encode("utf-8"))

    # some sugar to be used with "with stats:" at the beginning of the epoch
    def __enter__(self):
        if self.do_plot and self.epoch >= 0:
            self.plot_stats(self.visdom_env)
        self.new_epoch()

    def __exit__(self, type, value, traceback):
        iserr = type is not None and issubclass(type, Exception)
        iserr = iserr or (type is KeyboardInterrupt)
        if iserr:
            logger.error("error inside 'with' block")
            return
        if self.do_plot:
            self.plot_stats(self.visdom_env)

    def reset(self):  # to be called after each epoch
        stat_sets = list(self.stats.keys())
        logger.debug(f"stats: epoch {self.epoch} - reset")
        self.it = {k: -1 for k in stat_sets}
        for stat_set in stat_sets:
            for stat in self.stats[stat_set]:
                self.stats[stat_set][stat].reset()

    def hard_reset(self, epoch=-1):  # to be called during object __init__
        self.epoch = epoch
        logger.debug(f"stats: epoch {self.epoch} - hard reset")
        self.stats = {}

        # reset
        self.reset()

    def new_epoch(self):
        logger.debug(f"stats: new epoch {(self.epoch + 1)}")
        self.epoch += 1
        self.reset()  # zero the stats + increase epoch counter

    def gather_value(self, val):
        if isinstance(val, (float, int)):
            val = float(val)
        else:
            val = val.data.cpu().numpy()
            val = float(val.sum())
        return val

    def add_log_vars(self, added_log_vars):
        for add_log_var in added_log_vars:
            if add_log_var not in self.stats:
                logger.debug(f"Adding {add_log_var}")
                self.log_vars.append(add_log_var)

    def update(self, preds, time_start=None, freeze_iter=False, stat_set="train"):
        if self.epoch == -1:  # uninitialized
            logger.warning(
                "epoch==-1 means uninitialized stats structure -> new_epoch() called"
            )
            self.new_epoch()

        if stat_set not in self.stats:
            self.stats[stat_set] = {}
            self.it[stat_set] = -1

        if not freeze_iter:
            self.it[stat_set] += 1

        epoch = self.epoch

        for stat in self.log_vars:
            if stat not in self.stats[stat_set]:
                self.stats[stat_set][stat] = AverageMeter()

            if stat == "sec/it":  # compute speed
                if time_start is None:
                    time_per_it = 0.0
                else:
                    now = time.time()
                    time_per_it = now - (self._t_last_update or time_start)
                    self._t_last_update = now
                val = time_per_it
            else:
                if stat in preds:
                    try:
                        val = self.gather_value(preds[stat])
                    except KeyError:
                        raise ValueError(
                            "could not extract prediction %s\
                                          from the prediction dictionary"
                            % stat
                        ) from None
                else:
                    val = None

            if val is not None:
                self.stats[stat_set][stat].update(val, epoch=epoch, n=1)

    def get_epoch_averages(self, epoch=None):
        stat_sets = list(self.stats.keys())

        if epoch is None:
            epoch = self.epoch
        if epoch == -1:
            epoch = list(range(self.epoch))

        outvals = {}
        for stat_set in stat_sets:
            outvals[stat_set] = {
                "epoch": epoch,
                "it": self.it[stat_set],
                "epoch_max": self.epoch,
            }

            for stat in self.stats[stat_set].keys():
                if self.stats[stat_set][stat].count == 0:
                    continue
                if isinstance(epoch, Iterable):
                    avgs = self.stats[stat_set][stat].get_epoch_averages()
                    avgs = [avgs[e] for e in epoch]
                else:
                    avgs = self.stats[stat_set][stat].get_epoch_averages(epoch=epoch)
                outvals[stat_set][stat] = avgs

        return outvals

    def print(
        self,
        max_it=None,
        stat_set="train",
        vars_print=None,
        get_str=False,
        skip_nan=False,
        stat_format=lambda s: s.replace("loss_", "").replace("prev_stage_", "ps_"),
    ):
        """
        stats.print() is deprecated. Please use get_status_string() instead.
        example:
        std_out = stats.get_status_string()
        logger.info(str_out)
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
            if skip_nan and not np.isfinite(stats[stat_set][stat].avg):
                continue
            stat_str += " {0:.12}: {1:1.3f} |".format(
                stat_format(stat), stats[stat_set][stat].avg
            )

        head_str = "[%s] | epoch %3d | it %5d" % (stat_set, epoch, it)
        if max_it:
            head_str += "/ %d" % max_it

        str_out = "%s | %s" % (head_str, stat_str)

        if get_str:
            return str_out
        else:
            warnings.warn(
                "get_str=False is deprecated."
                "Please enable this flag to get receive the output string.",
                DeprecationWarning,
            )
            print(str_out)

    def get_status_string(
        self,
        max_it=None,
        stat_set="train",
        vars_print=None,
        skip_nan=False,
        stat_format=lambda s: s.replace("loss_", "").replace("prev_stage_", "ps_"),
    ):
        return self.print(
            max_it=max_it,
            stat_set=stat_set,
            vars_print=vars_print,
            get_str=True,
            skip_nan=skip_nan,
            stat_format=stat_format,
        )

    def plot_stats(
        self, visdom_env=None, plot_file=None, visdom_server=None, visdom_port=None
    ):
        # use the cached visdom env if none supplied
        if visdom_env is None:
            visdom_env = self.visdom_env
        if visdom_server is None:
            visdom_server = self.visdom_server
        if visdom_port is None:
            visdom_port = self.visdom_port
        if plot_file is None:
            plot_file = self.plot_file

        stat_sets = list(self.stats.keys())

        logger.debug(
            f"printing charts to visdom env '{visdom_env}' ({visdom_server}:{visdom_port})"
        )

        novisdom = False

        viz = get_visdom_connection(server=visdom_server, port=visdom_port)
        if viz is None or not viz.check_connection():
            logger.info("no visdom server! -> skipping visdom plots")
            novisdom = True

        lines = []

        # plot metrics
        if not novisdom:
            viz.close(env=visdom_env, win=None)

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

            lines.append((stat_sets_now, stat, vals))

        if not novisdom:
            for tmodes, stat, vals in lines:
                title = "%s" % stat
                opts = {"title": title, "legend": list(tmodes)}
                for i, (tmode, val) in enumerate(zip(tmodes, vals)):
                    update = "append" if i > 0 else None
                    valid = np.where(np.isfinite(val))[0]
                    if len(valid) == 0:
                        continue
                    x = np.arange(len(val))
                    viz.line(
                        Y=val[valid],
                        X=x[valid],
                        env=visdom_env,
                        opts=opts,
                        win=f"stat_plot_{title}",
                        name=tmode,
                        update=update,
                    )

        if plot_file:
            logger.info(f"plotting stats to {plot_file}")
            ncol = 3
            nrow = int(np.ceil(float(len(lines)) / ncol))
            matplotlib.rcParams.update({"font.size": 5})
            color = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
            fig = plt.figure(1)
            plt.clf()
            for idx, (tmodes, stat, vals) in enumerate(lines):
                c = next(color)
                plt.subplot(nrow, ncol, idx + 1)
                plt.gca()
                for vali, vals_ in enumerate(vals):
                    c_ = c * (1.0 - float(vali) * 0.3)
                    valid = np.where(np.isfinite(vals_))[0]
                    if len(valid) == 0:
                        continue
                    x = np.arange(len(vals_))
                    plt.plot(x[valid], vals_[valid], c=c_, linewidth=1)
                plt.ylabel(stat)
                plt.xlabel("epoch")
                plt.gca().yaxis.label.set_color(c[0:3] * 0.75)
                plt.legend(tmodes)
                gcolor = np.array(mcolors.to_rgba("lightgray"))
                grid_params = {"visible": True, "color": gcolor}
                plt.grid(**grid_params, which="major", linestyle="-", linewidth=0.4)
                plt.grid(**grid_params, which="minor", linestyle="--", linewidth=0.2)
                plt.minorticks_on()

            plt.tight_layout()
            plt.show()
            try:
                fig.savefig(plot_file)
            except PermissionError:
                warnings.warn("Cant dump stats due to insufficient permissions!")

    def synchronize_logged_vars(self, log_vars, default_val=float("NaN")):
        stat_sets = list(self.stats.keys())

        # remove the additional log_vars
        for stat_set in stat_sets:
            for stat in self.stats[stat_set].keys():
                if stat not in log_vars:
                    logger.warning(f"additional stat {stat_set}:{stat} -> removing")

            self.stats[stat_set] = {
                stat: v for stat, v in self.stats[stat_set].items() if stat in log_vars
            }

        self.log_vars = log_vars  # !!!

        for stat_set in stat_sets:
            for stat in log_vars:
                if stat not in self.stats[stat_set]:
                    logger.info(
                        "missing stat %s:%s -> filling with default values (%1.2f)"
                        % (stat_set, stat, default_val)
                    )
                elif len(self.stats[stat_set][stat].history) != self.epoch + 1:
                    h = self.stats[stat_set][stat].history
                    if len(h) == 0:  # just never updated stat ... skip
                        continue
                    else:
                        logger.info(
                            "incomplete stat %s:%s -> reseting with default values (%1.2f)"
                            % (stat_set, stat, default_val)
                        )
                else:
                    continue

                self.stats[stat_set][stat] = AverageMeter()
                self.stats[stat_set][stat].reset()

                lastep = self.epoch + 1
                for ep in range(lastep):
                    self.stats[stat_set][stat].update(default_val, n=1, epoch=ep)
                epoch_generated = self.stats[stat_set][stat].get_epoch()
                assert epoch_generated == self.epoch + 1, (
                    "bad epoch of synchronized log_var! %d vs %d"
                    % (
                        self.epoch + 1,
                        epoch_generated,
                    )
                )


class StatsJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (AverageMeter, Stats)):
            enc = self.encode(o.__dict__)
            return enc
        else:
            raise TypeError(
                f"Object of type {o.__class__.__name__} " f"is not JSON serializable"
            )


def _get_postfixed_filename(fl, postfix):
    return fl if fl.endswith(postfix) else fl + postfix
