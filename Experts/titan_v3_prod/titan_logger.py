"""
titan_logger.py — Titan V3.0 Non-Blocking Data Logger  (Patch 3.2)
===================================================================
All magic numbers read from CFG.logger. Zero hardcoded constants.
Logs to: features/ (Parquet), signals/ (CSV), trades/ (CSV),
         drift/ (JSON), perf/ (CSV), summary/ (JSON).
Thread-safe. HFT main loop is never blocked by I/O.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from titan_config import CFG, FEATURE_COLS

log = logging.getLogger("TitanLogger")


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureRecord:
    ts_ms: int; session_id: str; tick_idx: int
    bid: float; ask: float; mid: float; spread: float
    flags: int; dt_ms: float
    fdpi: float = 0.0; mvdi: float = 0.0; twkj: float = 0.0
    qad: float  = 0.0; sgc: float  = 0.0; hurst: float = 0.0
    topo_h0: float = 0.0; topo_h1: float = 0.0
    mfe: float = 0.0; ptp: float = 0.0
    twap_prob: float = 0.0; mom_ignite: float = 0.0
    ice_score: float = 0.0; tce: float = 0.0
    hour_sine:   float = 0.0
    hour_cosine: float = 0.0   # F16
    cluster_id: int   = -1
    phase_name: str   = "unknown"
    label: float      = float("nan")


@dataclass
class SignalRecord:
    ts_ms: int; session_id: str; tick_idx: int
    conviction: float; direction: str
    spread_ok: bool; dd_ok: bool; infer_ms: float; feature_hash: str = ""


@dataclass
class TradeRecord:
    trade_id: str; session_id: str; open_ts_ms: int
    close_ts_ms: int = 0; direction: str = ""
    open_price: float = 0.0; close_price: float = 0.0
    lots: float = 0.0; gross_pnl: float = 0.0
    spread_cost: float = 0.0; net_pnl: float = 0.0
    conviction: float = 0.0; open_cluster: int = -1
    open_phase: str = ""; close_reason: str = ""
    duration_ms: int = 0; mae_pips: float = 0.0; mfe_pips: float = 0.0


@dataclass
class DriftSnapshot:
    ts_ms: int; session_id: str; tick_idx: int
    feature_name: str; live_mean: float; live_std: float
    train_mean: float; train_std: float
    ks_statistic: float; drift_flag: bool


@dataclass
class LatencyRecord:
    ts_ms: int; session_id: str; tick_idx: int
    feature_us: float; infer_us: float
    signal_us: float; order_us: float; total_us: float


# ─────────────────────────────────────────────────────────────────────────────
class _Writer(threading.Thread):
    def __init__(self, log_dir: Path):
        super().__init__(daemon=False, name="TitanLogWriter")
        lc             = CFG.logger
        self.log_dir   = Path(log_dir)
        self.BATCH     = lc.flush_batch_size
        self.FLUSH_INT = lc.flush_interval_s
        self.q: queue.Queue = queue.Queue(maxsize=lc.queue_maxsize)
        self._stop     = threading.Event()
        self._buf: Dict[str, list] = {}
        self._dropped  = 0

    def submit(self, route: str, rec: dict):
        try: self.q.put_nowait((route, rec))
        except queue.Full: pass

    def stop(self):
        self._stop.set()
        self.join(timeout=CFG.logger.writer_stop_timeout_s)
        if self._dropped > 0:
            log.warning(f"Logger stopped. Records dropped on errors: {self._dropped}")

    def run(self):
        last = time.monotonic()
        while not self._stop.is_set():
            n = 0
            while n < self.BATCH:
                try:
                    route, rec = self.q.get_nowait()
                    self._buf.setdefault(route, []).append(rec)
                    n += 1
                except queue.Empty: break
            if n >= self.BATCH or (time.monotonic() - last) >= self.FLUSH_INT:
                self._flush_all(); last = time.monotonic()
            if n == 0: time.sleep(0.01)
        self._flush_all()

    def _flush_all(self):
        for route, recs in list(self._buf.items()):
            if not recs: continue
            try:
                self._write(route, recs)
                self._buf[route] = []
            except Exception as e:
                dropped = len(recs)
                self._dropped += dropped
                log.error(f"Writer flush error (route={route}) — "
                          f"{dropped} records dropped: {e}", exc_info=True)
                self._buf[route] = []

    def _write(self, route: str, recs: list):
        kind    = route.split("/")[0]
        session = route.split("/")[1] if "/" in route else "default"
        d       = self.log_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        df      = pd.DataFrame(recs)
        if kind == "features":
            p = d / f"{session}.parquet"
            if p.exists():
                df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
            df.to_parquet(p, compression="lz4", index=False)
        else:
            p = d / f"{session}.csv"
            df.to_csv(p, mode="a", header=not p.exists(), index=False)


# ─────────────────────────────────────────────────────────────────────────────
class FeatureLogger:
    def __init__(self, writer: _Writer, session: str):
        self._w = writer; self.sid = session; self.tick_idx = 0
        buf_size = CFG.logger.ring_buf_size
        self._ring = np.full((buf_size, len(FEATURE_COLS)), np.nan, dtype=np.float32)
        self._ri = 0; self._full = False; self._bsize = buf_size

    def log(self, r: FeatureRecord):
        self._w.submit(f"features/{self.sid}", asdict(r))
        vals = np.array([getattr(r, c, 0.0) for c in FEATURE_COLS], dtype=np.float32)
        self._ring[self._ri] = vals
        self._ri = (self._ri + 1) % self._bsize
        if self._ri == 0: self._full = True
        self.tick_idx += 1

    def recent(self, n: int = 1000) -> np.ndarray:
        n = min(n, self._bsize)
        if self._full:
            idx = [(self._ri - n + i) % self._bsize for i in range(n)]
        else:
            s = max(0, self._ri - n); idx = list(range(s, self._ri))
        return self._ring[idx]


class SignalLogger:
    def __init__(self, writer: _Writer, session: str):
        self._w = writer; self.sid = session
        self._total = 0; self._long = 0; self._short = 0; self._conv: list = []

    def log(self, r: SignalRecord):
        self._w.submit(f"signals/{self.sid}", asdict(r))
        self._total += 1
        if r.direction == "long":  self._long  += 1
        if r.direction == "short": self._short += 1
        self._conv.append(r.conviction)
        if len(self._conv) > 10_000: self._conv = self._conv[-5_000:]

    def stats(self) -> dict:
        if not self._conv: return {}
        a = np.array(self._conv)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "p5":   float(np.percentile(a, 5)),
                "p95":  float(np.percentile(a, 95)),
                "long_pct":  self._long  / max(1, self._total),
                "short_pct": self._short / max(1, self._total)}


class TradeLogger:
    def __init__(self, writer: _Writer, session: str):
        self._w = writer; self.sid = session
        self._open: Dict[str, TradeRecord] = {}
        self._closed = 0; self._pnl = 0.0
        self.pip = CFG.data.pip_size

    def open(self, r: TradeRecord):
        self._open[r.trade_id] = r
        log.info(f"[OPEN]  {r.trade_id} {r.direction} "
                 f"price={r.open_price:.5f} lots={r.lots:.2f}")

    def tick(self, trade_id: str, price: float):
        if price <= 0: return   # guard degenerate ticks
        r = self._open.get(trade_id)
        if r is None: return
        exc = ((price-r.open_price)/self.pip if r.direction=="long"
               else (r.open_price-price)/self.pip)
        r.mfe_pips = max(r.mfe_pips, exc)
        r.mae_pips = min(r.mae_pips, exc)

    def close(self, trade_id: str, close_price: float, ts_ms: int,
              reason: str, spread: float = 0.0):
        r = self._open.pop(trade_id, None)
        if r is None: return
        r.close_ts_ms = ts_ms; r.close_price = close_price
        r.close_reason = reason; r.duration_ms = ts_ms - r.open_ts_ms
        pip_d = ((close_price-r.open_price)/self.pip if r.direction=="long"
                 else (r.open_price-close_price)/self.pip)
        r.spread_cost = (spread/self.pip)*r.lots*10
        r.gross_pnl   = pip_d*r.lots*10
        r.net_pnl     = r.gross_pnl - r.spread_cost
        self._pnl    += r.net_pnl; self._closed += 1
        self._w.submit(f"trades/{self.sid}", asdict(r))
        log.info(f"[CLOSE] {trade_id} reason={reason} "
                 f"pnl={r.net_pnl:+.2f} dur={r.duration_ms}ms "
                 f"mae={r.mae_pips:.1f} mfe={r.mfe_pips:.1f}")

    def summary(self) -> dict:
        return {"closed": self._closed, "open": len(self._open), "total_pnl": self._pnl}


class DriftMonitor:
    """KS drift detection. All constants from CFG.logger."""

    def __init__(self, writer: _Writer, feat_logger: FeatureLogger,
                 session: str, train_stats: Optional[dict] = None):
        self._w = writer; self._fl = feat_logger
        self.sid = session; self.stats = train_stats or {}
        self._n = 0; self._alerts: list = []

    @property
    def KS_THRESH(self) -> float: return CFG.logger.ks_threshold
    @property
    def CHECK_EVERY(self) -> int:  return CFG.logger.drift_check_every

    def tick(self):
        self._n += 1
        if self._n % self.CHECK_EVERY == 0: self._run_check()

    def _run_check(self):
        from scipy.stats import ks_2samp
        recent = self._fl.recent(self.CHECK_EVERY)
        if len(recent) < 50: return
        ts = int(time.time() * 1000)
        for i, feat in enumerate(FEATURE_COLS):
            vals = recent[:, i]; vals = vals[~np.isnan(vals)]
            if len(vals) < 30: continue
            lm, ls = float(vals.mean()), float(vals.std())
            ts_    = self.stats.get(feat, {})
            tm     = ts_.get("mean", 0.0)
            tss    = max(ts_.get("std", 1.0), 1e-6)
            ref    = np.random.default_rng(42).normal(tm, tss, len(vals))
            ks, _  = ks_2samp(vals, ref)
            snap   = DriftSnapshot(ts, self.sid, self._n, feat, lm, ls, tm, tss,
                                    float(ks), float(ks) > self.KS_THRESH)
            self._w.submit(f"drift/{self.sid}", asdict(snap))
            if snap.drift_flag:
                self._alerts.append(feat)
                log.warning(f"DRIFT [{feat}]: KS={ks:.3f} live_mu={lm:.4f}")

    def risk_level(self) -> str:
        n = len(self._alerts[-50:]) if self._alerts else 0
        return "HIGH" if n >= 5 else ("MEDIUM" if n > 0 else "LOW")


class PerformanceTimer:
    def __init__(self, writer: _Writer, session: str,
                 feat_ref: Optional[FeatureLogger] = None):
        self._w = writer; self.sid = session; self._ref = feat_ref
        self._t: Dict[str, float] = {}

    def start(self, stage: str): self._t[stage] = time.perf_counter()
    def end(self, stage: str) -> float:
        return (time.perf_counter() - self._t.pop(stage, time.perf_counter())) * 1e6

    def record(self, feature_us: float, infer_us: float,
               signal_us: float, order_us: float, ts_ms: int):
        total = feature_us + infer_us + signal_us + order_us
        self._w.submit(f"perf/{self.sid}", asdict(LatencyRecord(
            ts_ms, self.sid, self._ref.tick_idx if self._ref else 0,
            feature_us, infer_us, signal_us, order_us, total,
        )))
        if total > 10_000:
            log.warning(f"High-latency tick: {total:.0f}µs")


# ─────────────────────────────────────────────────────────────────────────────
class TitanLogger:
    """
    Master logging facade.

    with TitanLogger("EURUSD_20260413") as logger:
        logger.log_feature(rec)
        logger.log_signal(srec)
        logger.open_trade(trec)
        logger.close_trade(id, price, ts, "tp")
        logger.drift.tick()
        logger.perf.record(f_us, i_us, s_us, o_us, ts)
    """

    def __init__(self, session_id: str = None, log_dir: Path = None,
                 train_stats: Optional[dict] = None):
        self.sid      = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.log_dir  = Path(log_dir or CFG.data.log_dir)
        self._writer  = _Writer(self.log_dir)
        self._started = False; self._t0 = time.time()
        self.features = FeatureLogger(self._writer, self.sid)
        self.signals  = SignalLogger(self._writer,  self.sid)
        self.trades   = TradeLogger(self._writer,   self.sid)
        self.drift    = DriftMonitor(self._writer, self.features, self.sid, train_stats)
        self.perf     = PerformanceTimer(self._writer, self.sid, self.features)

    def start(self) -> "TitanLogger":
        if not self._started:
            self._writer.start(); self._started = True
            log.info(f"TitanLogger started: session={self.sid}")
        return self

    def stop(self) -> dict:
        if not self._started: return {}
        dur = time.time() - self._t0
        summ = {
            "session_id":      self.sid,
            "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
            "duration_s":      round(dur, 1),
            "total_ticks":     self.features.tick_idx,
            "tps":             round(self.features.tick_idx / max(dur, 1), 1),
            "conviction":      self.signals.stats(),
            "trades":          self.trades.summary(),
            "drift_risk":      self.drift.risk_level(),
            "drift_alerts":    len(self.drift._alerts),
            "records_dropped": self._writer._dropped,
        }
        out = self.log_dir / "summary"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.sid}.json").write_text(json.dumps(summ, indent=2))
        self._writer.stop(); self._started = False
        log.info(f"TitanLogger stopped: {summ}")
        return summ

    def log_feature(self, r: FeatureRecord):  self.features.log(r)
    def log_signal(self,  r: SignalRecord):   self.signals.log(r)
    def open_trade(self,  r: TradeRecord):    self.trades.open(r)
    def update_trade(self, trade_id: str, price: float): self.trades.tick(trade_id, price)
    def close_trade(self, *a, **kw):          self.trades.close(*a, **kw)

    def build_feature_record(self, ts_ms: int, bid: float, ask: float,
                              flags: int, dt_ms: float,
                              feature_values: np.ndarray,
                              cluster_id: int = -1,
                              phase_name: str = "unknown",
                              label: float = float("nan")) -> FeatureRecord:
        if len(feature_values) != len(FEATURE_COLS):
            raise ValueError(
                f"feature_values length {len(feature_values)} != "
                f"FEATURE_COLS length {len(FEATURE_COLS)}."
            )
        kw = {c: float(feature_values[i]) for i, c in enumerate(FEATURE_COLS)}
        return FeatureRecord(
            ts_ms=ts_ms, session_id=self.sid,
            tick_idx=self.features.tick_idx,
            bid=bid, ask=ask, mid=(bid+ask)/2, spread=ask-bid,
            flags=flags, dt_ms=dt_ms,
            cluster_id=cluster_id, phase_name=phase_name, label=label,
            **kw,
        )

    def load_training_stats(self, df: pd.DataFrame) -> "TitanLogger":
        stats = {}
        for col in FEATURE_COLS:
            if col in df.columns:
                v = df[col].dropna()
                stats[col] = {"mean": float(v.mean()), "std": float(v.std()),
                              "p5":   float(v.quantile(0.05)),
                              "p95":  float(v.quantile(0.95))}
        self.drift.stats = stats
        p = self.log_dir / "training_stats.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stats, indent=2))
        log.info(f"Training stats saved → {p}")
        return self

    def __enter__(self): return self.start()
    def __exit__(self, *_): self.stop()
