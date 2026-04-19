"""
titan_mt5_bridge.py — Titan V3.0 MT5 Python Bridge
====================================================
Fixes v3.1:
  - LiveTickStreamer uses daemon=False for clean shutdown
  - HistoricalDownloader passes datetime objects to copy_ticks_range (not Unix ms)
  - _require_windows() called in connect(), not just __init__
  - reconnect() method added to handle disconnection
  - AccountMonitor polls at 5s (not 1s) to avoid blocking COM calls
  - MT5TradeExecutor has ORDER_FILLING_RETURN fallback
"""
from __future__ import annotations

import logging
import platform
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from titan_config import CFG
from titan_data import FlagMapper, ParquetConverter
from titan_logger import TitanLogger, FeatureRecord

log = logging.getLogger("TitanMT5Bridge")


def _require_windows():
    if platform.system() != "Windows":
        raise OSError(
            "MetaTrader5 Python API requires Windows.\n"
            f"Current OS: {platform.system()}\n"
            "Options:\n"
            "  1. Run on Windows with MT5 terminal installed and logged in.\n"
            "  2. Use titan_data.ParquetConverter to load pre-exported files."
        )


class _MT5:
    _pkg = None
    @classmethod
    def get(cls):
        if cls._pkg is None:
            try:
                import MetaTrader5 as mt5
                cls._pkg = mt5
            except ImportError:
                raise ImportError(
                    "MetaTrader5 not installed.\n"
                    "Install: pip install MetaTrader5\n"
                    "Requires: Windows + MT5 terminal running."
                )
        return cls._pkg


@dataclass
class MT5Tick:
    time_ms: int; bid: float; ask: float; flags: int
    last: float = 0.0; volume: int = 0

    def to_row(self) -> dict:
        return {"Tick_Time_ms": self.time_ms, "Bid": self.bid,
                "Ask": self.ask, "Flags": self.flags}


@dataclass
class AccountState:
    equity: float; balance: float; margin: float
    free_margin: float; margin_level: float
    profit: float; timestamp: int


# ─────────────────────────────────────────────────────────────────────────────
class HistoricalDownloader:
    CHUNK = 7  # days

    def __init__(self, symbol: str, out_dir: Path = None):
        self.symbol  = symbol
        self.out_dir = Path(out_dir or CFG.data.parquet_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mt5     = _MT5.get()

    def fetch(self, days: int = 30, start: Optional[datetime] = None,
              end: Optional[datetime] = None) -> List[Path]:
        utc = timezone.utc
        if end   is None: end   = datetime.now(utc)
        if start is None: start = end - timedelta(days=days)
        log.info(f"Downloading {self.symbol}: {start.date()} → {end.date()}")
        paths, cur = [], start
        while cur < end:
            ch_end = min(cur + timedelta(days=self.CHUNK), end)
            p = self._chunk(cur, ch_end)
            if p: paths.append(p)
            cur = ch_end
        log.info(f"Download complete: {len(paths)} files")
        return paths

    def _chunk(self, s: datetime, e: datetime) -> Optional[Path]:
        # Fix: pass datetime objects, NOT Unix milliseconds.
        # copy_ticks_range expects naive UTC datetime.
        s_naive = s.replace(tzinfo=None)
        e_naive = e.replace(tzinfo=None)
        raw = self.mt5.copy_ticks_range(
            self.symbol, s_naive, e_naive, self.mt5.COPY_TICKS_ALL
        )
        if raw is None or len(raw) == 0:
            log.warning(f"No ticks: {s.date()} → {e.date()}")
            return None
        log.info(f"  {s.date()}: {len(raw):,} ticks")
        df  = self._normalise(pd.DataFrame(raw))
        fn  = f"{self.symbol}_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}.parquet"
        out = self.out_dir / fn
        df.to_parquet(out, compression="lz4", index=False)
        log.info(f"  Saved {out.name} ({out.stat().st_size/1e6:.1f} MB)")
        return out

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        if "time_msc" in df.columns:
            out["Tick_Time_ms"] = df["time_msc"].astype(np.int64)
        elif "time" in df.columns:
            out["Tick_Time_ms"] = (df["time"] * 1000).astype(np.int64)
        else:
            raise ValueError("No time column in MT5 tick data.")
        out["Bid"]   = df["bid"].astype(np.float64)
        out["Ask"]   = df["ask"].astype(np.float64)
        out["Flags"] = FlagMapper.translate_series(
            pd.to_numeric(df.get("flags", 6), errors="coerce").fillna(6)
        ) if "flags" in df.columns else 6
        out = out[(out["Bid"]>0)&(out["Ask"]>0)&(out["Ask"]>=out["Bid"])]
        out = out.drop_duplicates("Tick_Time_ms")
        return out.sort_values("Tick_Time_ms").reset_index(drop=True)

    def fetch_latest_n(self, n: int = 100_000) -> pd.DataFrame:
        # Pass datetime object, not Unix ms
        from_dt = datetime.utcnow() - timedelta(hours=48)
        raw = self.mt5.copy_ticks_from(
            self.symbol, from_dt, n, self.mt5.COPY_TICKS_ALL
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"Failed to fetch latest {n} ticks.")
        return self._normalise(pd.DataFrame(raw)).tail(n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
class LiveTickStreamer:
    """
    Polls MT5 at ~200Hz max.
    daemon=False — ensures flush completes cleanly on shutdown.
    """

    def __init__(self, symbol: str, poll_ms: int = 5):
        self.symbol      = symbol
        self.poll_ms     = poll_ms
        self.mt5         = _MT5.get()
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._cbs: List[Callable] = []
        self._last_ms    = 0
        self.tick_count  = 0

    def add_callback(self, fn: Callable) -> "LiveTickStreamer":
        self._cbs.append(fn); return self

    def start(self, logger: Optional[TitanLogger] = None) -> "LiveTickStreamer":
        if self._running: return self
        if logger: self.add_callback(self._logger_cb(logger))
        self._running = True
        self._last_ms = int(time.time() * 1000) - self.poll_ms
        self._thread  = threading.Thread(
            target=self._loop, daemon=False, name="MT5Poller"   # daemon=False
        )
        self._thread.start()
        log.info(f"LiveTickStreamer started: {self.symbol} @ {self.poll_ms}ms")
        return self

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=5.0)
        log.info(f"LiveTickStreamer stopped: {self.tick_count:,} ticks.")

    def _loop(self):
        while self._running:
            t0 = time.perf_counter()
            try: self._fetch()
            except Exception as e: log.error(f"Tick poll: {e}")
            sl = max(0.0, self.poll_ms/1000 - (time.perf_counter()-t0))
            if sl > 0: time.sleep(sl)

    def _fetch(self):
        from_dt = datetime.utcfromtimestamp(self._last_ms / 1000.0)
        raw     = self.mt5.copy_ticks_from(
            self.symbol, from_dt, 1000, self.mt5.COPY_TICKS_ALL
        )
        if raw is None or len(raw) == 0: return
        for row in raw:
            ms = int(row["time_msc"])
            if ms <= self._last_ms: continue
            tick = MT5Tick(
                time_ms=ms, bid=float(row["bid"]), ask=float(row["ask"]),
                flags=FlagMapper.translate(int(row.get("flags", 6))),
                last=float(row.get("last", 0.0)), volume=int(row.get("volume", 0)),
            )
            for cb in self._cbs:
                try: cb(tick)
                except Exception as e: log.error(f"Tick callback: {e}")
            self.tick_count += 1; self._last_ms = ms

    def _logger_cb(self, logger: TitanLogger) -> Callable:
        prev = [int(time.time() * 1000)]
        def _cb(tick: MT5Tick):
            dt = max(tick.time_ms - prev[0], CFG.features.min_dt_ms)
            prev[0] = tick.time_ms
            rec = FeatureRecord(
                ts_ms=tick.time_ms, session_id=logger.sid,
                tick_idx=logger.features.tick_idx,
                bid=tick.bid, ask=tick.ask, mid=(tick.bid+tick.ask)/2,
                spread=tick.ask-tick.bid, flags=tick.flags, dt_ms=dt,
            )
            logger.log_feature(rec)
        return _cb


# ─────────────────────────────────────────────────────────────────────────────
class AccountMonitor:
    """
    Polls account state. Interval = 5s (not 1s).
    COM calls on Windows can block for 50-200ms; 5s interval keeps CPU low.
    """

    def __init__(self, poll_s: float = 5.0):   # 5s default (was 1s)
        self.poll_s   = poll_s
        self.mt5      = _MT5.get()
        self._state: Optional[AccountState] = None
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._nav_start = 0.0
        self._dd_cbs: List[Callable] = []

    def add_dd_callback(self, fn: Callable): self._dd_cbs.append(fn)

    def start(self) -> "AccountMonitor":
        if self._running: return self
        self._nav_start = self._equity()
        self._running   = True
        self._thread    = threading.Thread(
            target=self._loop, daemon=False, name="AccountMon"
        )
        self._thread.start()
        log.info(f"AccountMonitor started. NAV={self._nav_start:.2f} poll={self.poll_s}s")
        return self

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=10.0)

    def _equity(self) -> float:
        info = self.mt5.account_info()
        return float(info.equity) if info else 0.0

    def _loop(self):
        while self._running:
            try:
                info = self.mt5.account_info()
                if info:
                    self._state = AccountState(
                        equity=float(info.equity), balance=float(info.balance),
                        margin=float(info.margin), free_margin=float(info.margin_free),
                        margin_level=float(info.margin_level or 0),
                        profit=float(info.profit), timestamp=int(time.time()*1000),
                    )
                    if self._nav_start > 0:
                        dd = (self._state.equity - self._nav_start) / self._nav_start
                        if dd < -CFG.execution.daily_dd_limit:
                            log.critical(f"DD KILL-SWITCH: {dd:.2%}")
                            for fn in self._dd_cbs:
                                try: fn(dd, self._state)
                                except Exception: pass
            except Exception as e: log.error(f"AccountMonitor: {e}")
            time.sleep(self.poll_s)

    @property
    def state(self): return self._state
    def daily_dd(self) -> float:
        if not self._state or self._nav_start <= 0: return 0.0
        return (self._state.equity - self._nav_start) / self._nav_start


# ─────────────────────────────────────────────────────────────────────────────
class MT5TradeExecutor:
    """For paper trading and research only. Includes ORDER_FILLING_RETURN fallback."""

    def __init__(self, symbol: str, magic: int = 30000, dev: int = 10):
        self.symbol = symbol; self.magic = magic; self.dev = dev
        self.mt5    = _MT5.get()

    def _filling_type(self):
        """Return IOC if supported, else RETURN (broker-dependent)."""
        info = self.mt5.symbol_info(self.symbol)
        if info is None: return self.mt5.ORDER_FILLING_IOC
        mode = info.filling_mode
        if mode & self.mt5.SYMBOL_FILLING_IOC:   return self.mt5.ORDER_FILLING_IOC
        if mode & self.mt5.SYMBOL_FILLING_RETURN: return self.mt5.ORDER_FILLING_RETURN
        return self.mt5.ORDER_FILLING_IOC   # last resort

    def _send(self, req: dict) -> Optional[int]:
        req["type_filling"] = self._filling_type()
        r = self.mt5.order_send(req)
        if r.retcode != self.mt5.TRADE_RETCODE_DONE:
            log.error(f"Order failed: {r.retcode} {r.comment}")
            return None
        return r.order

    def buy(self, lots: float, comment: str = "TitanV3") -> Optional[int]:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None: return None
        return self._send({
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
            "volume": round(lots,2), "type": self.mt5.ORDER_TYPE_BUY,
            "price": tick.ask, "deviation": self.dev, "magic": self.magic,
            "comment": comment, "type_time": self.mt5.ORDER_TIME_GTC,
        })

    def sell(self, lots: float, comment: str = "TitanV3") -> Optional[int]:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None: return None
        return self._send({
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
            "volume": round(lots,2), "type": self.mt5.ORDER_TYPE_SELL,
            "price": tick.bid, "deviation": self.dev, "magic": self.magic,
            "comment": comment, "type_time": self.mt5.ORDER_TIME_GTC,
        })

    def close(self, ticket: int) -> bool:
        pos  = self.mt5.positions_get(ticket=ticket)
        if not pos: return False
        p    = pos[0]
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None: return False
        ct   = self.mt5.ORDER_TYPE_SELL if p.type==self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY
        cp   = tick.bid if ct==self.mt5.ORDER_TYPE_SELL else tick.ask
        ok   = self._send({
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
            "volume": p.volume, "type": ct, "position": ticket,
            "price": cp, "deviation": self.dev, "magic": self.magic,
            "comment": "TitanV3_close", "type_time": self.mt5.ORDER_TIME_GTC,
        }) is not None
        if ok: log.info(f"Closed {ticket} @ {cp}")
        return ok

    def close_all(self) -> int:
        pos = self.mt5.positions_get(symbol=self.symbol) or []
        n   = sum(1 for p in pos if p.magic==self.magic and self.close(p.ticket))
        log.info(f"Emergency close: {n} positions."); return n


# ─────────────────────────────────────────────────────────────────────────────
class MT5Bridge:
    """
    Master facade.
    _require_windows() called in connect() (not just __init__) —
    prevents bypass via subclassing.
    reconnect() added for automatic disconnection recovery.
    """

    def __init__(self, symbol: str = None, login: int = None,
                 password: str = None, server: str = None,
                 path: str = None, magic: int = 30000, poll_ms: int = 5):
        self.symbol  = symbol or CFG.data.symbol
        self._login  = login; self._pw = password
        self._srv    = server; self._path = path
        self.mt5     = _MT5.get()
        self._conn   = False
        self.downloader: Optional[HistoricalDownloader] = None
        self.streamer:   Optional[LiveTickStreamer]      = None
        self.account:    Optional[AccountMonitor]        = None
        self.executor:   Optional[MT5TradeExecutor]      = None
        self._poll_ms = poll_ms; self._magic = magic

    def connect(self) -> "MT5Bridge":
        _require_windows()   # always check here, not just __init__
        kw = {}
        if self._path:  kw["path"]     = self._path
        if self._login: kw["login"]    = self._login
        if self._pw:    kw["password"] = self._pw
        if self._srv:   kw["server"]   = self._srv
        if not self.mt5.initialize(**kw):
            raise ConnectionError(
                f"MT5 init failed: {self.mt5.last_error()}.  "
                "Ensure terminal is running and logged in."
            )
        acct = self.mt5.account_info()
        log.info(f"MT5 connected: "
                 f"account={acct.login if acct else '?'} "
                 f"broker={acct.company if acct else '?'}")
        self.mt5.symbol_select(self.symbol, True)
        self._conn      = True
        self.downloader = HistoricalDownloader(self.symbol)
        self.streamer   = LiveTickStreamer(self.symbol, self._poll_ms)
        self.account    = AccountMonitor()
        self.executor   = MT5TradeExecutor(self.symbol, self._magic)
        return self

    def reconnect(self) -> bool:
        """
        Detect and recover from MT5 disconnection.
        Returns True if reconnected successfully.
        """
        err_code = self.mt5.last_error()[0] if hasattr(self.mt5, "last_error") else 0
        # MT5 error -10004 = disconnected
        if err_code == -10004 or not self._conn:
            log.warning("MT5 disconnected — attempting reconnect…")
            try:
                self.disconnect()
                time.sleep(2.0)
                self.connect()
                log.info("MT5 reconnected successfully.")
                return True
            except Exception as e:
                log.error(f"Reconnect failed: {e}")
                return False
        return True

    def disconnect(self):
        if self.streamer and self.streamer._running: self.streamer.stop()
        if self.account  and self.account._running:  self.account.stop()
        if self._conn:
            self.mt5.shutdown(); self._conn = False
            log.info("MT5 disconnected.")

    def status(self) -> dict:
        if not self._conn: return {"connected": False}
        acct = self.mt5.account_info()
        tick = self.mt5.symbol_info_tick(self.symbol)
        return {"connected": True, "symbol": self.symbol,
                "bid": tick.bid if tick else None,
                "ask": tick.ask if tick else None,
                "equity": acct.equity if acct else None}

    def __enter__(self): return self.connect()
    def __exit__(self, *_): self.disconnect()
