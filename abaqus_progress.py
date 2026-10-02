# -*- coding: utf-8 -*-
"""Aggregate runtime progress reporting for the Abaqus modal pipeline."""
from __future__ import print_function
import json
import math
import os
import time


def _fmt_seconds(value):
    if value is None or not math.isfinite(float(value)):
        return "?"
    seconds = max(0, int(round(float(value))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh%02dm%02ds" % (h, m, s)
    if m:
        return "%dm%02ds" % (m, s)
    return "%ds" % s


def solver_estimate_fraction(elapsed_seconds, expected_seconds=1800.0):
    """Monotonic bounded first-run estimate for stages without exact counters."""
    elapsed = max(0.0, float(elapsed_seconds))
    expected = max(1.0, float(expected_seconds or 1800.0))
    if elapsed <= expected:
        return min(0.90, 0.90 * elapsed / expected)
    # Past the nominal duration, approach 0.99 slowly but never claim completion.
    extra = elapsed - expected
    return min(0.99, 0.90 + 0.09 * extra / (extra + expected))


class ProgressTracker(object):
    def __init__(self, stage_names, stage_weights=None, emit=print, clock=time.time,
                 history_path=None, signature=None, observer=None):
        self.stage_names = list(stage_names)
        if not self.stage_names or len(set(self.stage_names)) != len(self.stage_names):
            raise ValueError("stage_names must be nonempty and unique")
        weights = [1.0] * len(self.stage_names) if stage_weights is None else list(stage_weights)
        if len(weights) != len(self.stage_names) or any((not math.isfinite(float(w)) or float(w) <= 0)
                                                        for w in weights):
            raise ValueError("stage_weights must be positive finite values")
        scale = sum(float(w) for w in weights)
        self.weights = [float(w) / scale for w in weights]
        self.emit = emit
        self.clock = clock
        self.history_path = history_path
        self.signature = signature
        self.observer = observer
        self.created_at = self.clock()
        self.stage = None
        self.stage_index = None
        self.stage_started = None
        self.stage_fraction = 0.0
        self.completed = set()
        self.stage_durations = {}
        self.stage_started_epochs = {}
        self.stage_finished_epochs = {}
        self.last = None

    def _remaining(self):
        if self.stage_index is None:
            return [s for i, s in enumerate(self.stage_names) if i not in self.completed]
        return [s for i, s in enumerate(self.stage_names)
                if i > self.stage_index and i not in self.completed]

    def _overall_fraction(self):
        done = sum(self.weights[i] for i in self.completed)
        if self.stage_index is not None and self.stage_index not in self.completed:
            done += self.weights[self.stage_index] * self.stage_fraction
        return min(1.0, max(0.0, done))

    def _row(self, now, estimated=False, done=None, total=None, rate=None, eta=None, note=None):
        row = dict(
            stage=self.stage,
            stage_percent=100.0 * self.stage_fraction,
            overall_percent=100.0 * self._overall_fraction(),
            estimated=bool(estimated),
            done=done,
            total=total,
            rate_per_second=rate,
            eta_seconds=eta,
            elapsed_seconds=max(0.0, now - self.created_at),
            stage_elapsed_seconds=(None if self.stage_started is None
                                   else max(0.0, now - self.stage_started)),
            remaining_stages=self._remaining(),
            note=note,
            stage_durations_seconds=dict(self.stage_durations))
        self.last = row
        return row

    def _format(self, row):
        prefix = "ESTIMATED " if row["estimated"] else ""
        units = ""
        if row["done"] is not None and row["total"] is not None:
            units = " %s/%s" % (row["done"], row["total"])
        rate = ""
        if row["rate_per_second"] is not None:
            rate = " rate=%.3g/s" % row["rate_per_second"]
        eta = " ETA=%s" % _fmt_seconds(row["eta_seconds"])
        remaining = ",".join(row["remaining_stages"]) or "none"
        note = (" | " + str(row["note"])) if row.get("note") else ""
        return ("PROGRESS %s| stage=%s %.1f%%%s | overall=%.1f%% | elapsed=%s%s%s | remaining=%s%s" %
                (prefix, row["stage"], row["stage_percent"], units,
                 row["overall_percent"], _fmt_seconds(row["elapsed_seconds"]),
                 rate, eta, remaining, note))

    def _publish(self, row):
        self.emit(self._format(row))
        if self.observer is not None:
            self.observer(dict(row))
        return row

    def start(self, stage):
        if stage not in self.stage_names:
            raise ValueError("Unknown stage: %s" % stage)
        index = self.stage_names.index(stage)
        if self.stage_index is not None and index < self.stage_index:
            raise ValueError("Progress stages cannot move backward")
        self.stage = stage
        self.stage_index = index
        self.stage_started = self.clock()
        self.stage_started_epochs[stage] = self.stage_started
        self.stage_fraction = 0.0
        row = self._row(self.stage_started)
        return self._publish(row)

    def update(self, done=None, total=None, estimate_fraction=None, note=None):
        if self.stage is None:
            raise RuntimeError("start(stage) must be called before update")
        now = self.clock()
        estimated = estimate_fraction is not None
        rate = eta = None
        if done is not None or total is not None:
            if done is None or total is None or float(total) <= 0:
                raise ValueError("done and positive total must be supplied together")
            fraction = min(1.0, max(0.0, float(done) / float(total)))
            self.stage_fraction = max(self.stage_fraction, fraction)
            elapsed = max(0.0, now - self.stage_started)
            if elapsed > 0 and float(done) > 0:
                rate = float(done) / elapsed
                eta = max(0.0, (float(total) - float(done)) / rate)
        elif estimate_fraction is not None:
            fraction = min(0.99, max(0.0, float(estimate_fraction)))
            self.stage_fraction = max(self.stage_fraction, fraction)
            elapsed = max(0.0, now - self.stage_started)
            if self.stage_fraction > 0:
                eta = elapsed * (1.0 - self.stage_fraction) / self.stage_fraction
        row = self._row(now, estimated=estimated, done=done, total=total,
                        rate=rate, eta=eta, note=note)
        return self._publish(row)

    def finish(self, stage=None, note=None):
        if self.stage is None:
            raise RuntimeError("No active stage")
        if stage is not None and stage != self.stage:
            raise ValueError("Active stage is %s, not %s" % (self.stage, stage))
        now = self.clock()
        self.stage_fraction = 1.0
        self.stage_durations[self.stage] = max(0.0, now - self.stage_started)
        self.stage_finished_epochs[self.stage] = now
        self.completed.add(self.stage_index)
        row = self._row(now, note=note)
        self._publish(row)
        self._save_history(now)
        return row

    def summary(self):
        now = self.clock()
        return self._row(now, estimated=bool(self.last and self.last.get("estimated")))

    def historical_seconds(self, stage, default=None):
        if not self.history_path or not self.signature or not os.path.isfile(self.history_path):
            return default
        try:
            with open(self.history_path, "r") as stream:
                data = json.load(stream)
            value = data.get(self.signature, {}).get(stage)
            return float(value) if value is not None else default
        except Exception:
            return default

    def _save_history(self, now):
        if not self.history_path or not self.signature or self.stage_started is None:
            return
        duration = max(0.0, now - self.stage_started)
        try:
            data = {}
            if os.path.isfile(self.history_path):
                with open(self.history_path, "r") as stream:
                    data = json.load(stream)
            record = data.setdefault(self.signature, {})
            old = record.get(self.stage)
            record[self.stage] = duration if old is None else 0.7 * float(old) + 0.3 * duration
            tmp = self.history_path + ".tmp"
            with open(tmp, "w") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
            os.replace(tmp, self.history_path)
        except Exception:
            pass
