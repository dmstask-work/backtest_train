"""
Modul Generator Sinyal Trading Adaptif (Fase 3) — v3.

Menghasilkan sinyal entry LONG (BUY +1), SHORT (SELL -1), atau HOLD (0)
berdasarkan kondisi pasar (regime) saat ini. Seluruh logika diterapkan secara
vectorized; tidak ada loop eksplisit.

Strategi yang Digunakan:
──────────────────────────────────────────────────────────────────────────────
  Regime TRENDING → Trend-Following

  ┌──────────────┬──────────────────────────────────────────────────────────┐
  │ LONG  (BUY)  │ EMA fast > EMA slow        [struktur tren naik]         │
  │              │ MACD histogram cross-up 0  [momentum naik dimulai]      │
  │              │ Supertrend direction = +1  [harga di atas supertrend]   │
  │              │ Volume > Volume SMA        [konfirmasi volume]           │
  ├──────────────┼──────────────────────────────────────────────────────────┤
  │ SHORT (SELL) │ EMA fast < EMA slow        [struktur tren turun]        │
  │              │ MACD histogram cross-down 0[momentum turun dimulai]     │
  │              │ Supertrend direction = −1  [harga di bawah supertrend]  │
  │              │ Volume > Volume SMA        [konfirmasi volume]           │
  └──────────────┴──────────────────────────────────────────────────────────┘

  Regime SIDEWAYS → Mean-Reversion

  ┌──────────────┬──────────────────────────────────────────────────────────┐
  │ LONG  (BUY)  │ Close ≤ BB Lower           [harga di tepi bawah]        │
  │              │ RSI < rsi_oversold  (30)   [kondisi oversold]           │
  │              │ BB Bandwidth < SMA × factor[BB tidak expanding]         │
  │              │ Volume > Volume SMA        [konfirmasi volume]           │
  ├──────────────┼──────────────────────────────────────────────────────────┤
  │ SHORT (SELL) │ Close ≥ BB Upper           [harga di tepi atas]         │
  │              │ RSI > rsi_overbought (70)  [kondisi overbought]         │
  │              │ BB Bandwidth < SMA × factor[BB tidak expanding]         │
  │              │ Volume > Volume SMA        [konfirmasi volume]           │
  └──────────────┴──────────────────────────────────────────────────────────┘

Strategy Switcher:
  enable_trend_following dan enable_mean_reversion dapat dimatikan secara
  independen melalui CLI atau settings.json untuk A/B testing.
  Saat dimatikan, kode strategi TIDAK dihapus — hanya di-skip saat evaluasi.
"""

import logging

import numpy as np
import pandas as pd

from config import INDICATOR_CONFIG
from regime_filter import REGIME_SIDEWAYS, REGIME_TRENDING

logger = logging.getLogger(__name__)

# ── Konstanta nilai sinyal ───────────────────────────────────────────────────
SIGNAL_BUY:  int =  1
SIGNAL_SELL: int = -1
SIGNAL_HOLD: int =  0

# ── Label strategi (digunakan untuk analitik breakdown di reporter) ──────────
STRATEGY_TREND_LONG:    str = "TREND_FOLLOWING_LONG"
STRATEGY_TREND_SHORT:   str = "TREND_FOLLOWING_SHORT"
STRATEGY_MEAN_REV_LONG:  str = "MEAN_REVERSION_LONG"
STRATEGY_MEAN_REV_SHORT: str = "MEAN_REVERSION_SHORT"
STRATEGY_NONE:          str = "NONE"

# Alias backward-compatible (optimizer.py / reporter.py lama)
STRATEGY_TREND:    str = STRATEGY_TREND_LONG
STRATEGY_MEAN_REV: str = STRATEGY_MEAN_REV_LONG


class SignalGenerator:
    """
    Menghasilkan sinyal LONG/SHORT adaptif berdasarkan regime pasar terdeteksi.

    Empat sub-strategi bekerja secara eksklusif sesuai kondisi:
      Trending  → Trend-Following LONG  / Trend-Following SHORT
      Sideways  → Mean-Reversion  LONG  / Mean-Reversion  SHORT

    Setiap strategi (trend / mean-reversion) dapat dimatikan secara independen
    via saklar enable_trend_following / enable_mean_reversion tanpa menghapus
    kodenya, sehingga mudah untuk A/B testing dan forward analysis.
    """

    def __init__(
        self,
        config: dict = INDICATOR_CONFIG,
        enable_trend_following: bool = True,
        enable_mean_reversion: bool = True,
    ) -> None:
        """
        Inisialisasi generator sinyal.

        Args:
            config:                  Dictionary parameter indikator.
            enable_trend_following:  Aktifkan strategi Trend-Following (long & short).
            enable_mean_reversion:   Aktifkan strategi Mean-Reversion (long & short).
        """
        self.cfg = config
        self.enable_trend_following = enable_trend_following
        self.enable_mean_reversion  = enable_mean_reversion

        if not enable_trend_following and not enable_mean_reversion:
            logger.warning(
                "[SignalGenerator] ⚠ Kedua strategi dinonaktifkan! "
                "Tidak ada sinyal LONG/SHORT yang akan dihasilkan."
            )

        logger.info(
            "[SignalGenerator] Inisialisasi | Trend-Following: %s | Mean-Reversion: %s",
            "ON" if enable_trend_following else "OFF",
            "ON" if enable_mean_reversion  else "OFF",
        )

    # ------------------------------------------------------------------
    # Fungsi Generate Sinyal
    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghasilkan sinyal trading untuk seluruh DataFrame secara vectorized.

        Args:
            df: DataFrame dengan kolom indikator lengkap dan kolom 'regime'.

        Returns:
            DataFrame dengan dua kolom tambahan:
              signal        (int): 1 = LONG, -1 = SHORT, 0 = HOLD
              strategy_used (str): Label sub-strategi yang memicu sinyal
        """
        self._validate_columns(df)
        df = df.copy()

        # ──────────────────────────────────────────────────────────────────
        # FILTER UNIVERSAL — Konfirmasi Volume (berlaku untuk semua sinyal)
        # Tolak sinyal jika volume candle ini < rata-rata volume (SMA Vol)
        # Tujuan: menyaring fake breakout yang digerakkan volume rendah
        # ──────────────────────────────────────────────────────────────────
        volume_confirmed: pd.Series = df["volume"] > df["volume_sma"]

        # ──────────────────────────────────────────────────────────────────
        # STRATEGI A — Trend-Following  (aktif saat TRENDING)
        # ──────────────────────────────────────────────────────────────────
        if self.enable_trend_following:
            macd_cross_up: pd.Series = (
                (df["macd_histogram"] > 0)
                & (df["macd_histogram"].shift(1) <= 0)
            )
            macd_cross_down: pd.Series = (
                (df["macd_histogram"] < 0)
                & (df["macd_histogram"].shift(1) >= 0)
            )

            # LONG: EMA bullish + MACD cross-up + Supertrend bullish
            trend_long: pd.Series = (
                (df["regime"] == REGIME_TRENDING)
                & (df["ema_fast"] > df["ema_slow"])
                & macd_cross_up
                & (df["supertrend_dir"] == 1)           # harga di atas supertrend
                & volume_confirmed
            )
            # SHORT: EMA bearish + MACD cross-down + Supertrend bearish
            trend_short: pd.Series = (
                (df["regime"] == REGIME_TRENDING)
                & (df["ema_fast"] < df["ema_slow"])
                & macd_cross_down
                & (df["supertrend_dir"] == -1)          # harga di bawah supertrend
                & volume_confirmed
            )
        else:
            trend_long  = pd.Series(False, index=df.index)
            trend_short = pd.Series(False, index=df.index)

        # ──────────────────────────────────────────────────────────────────
        # STRATEGI B — Mean-Reversion  (aktif saat SIDEWAYS)
        # ──────────────────────────────────────────────────────────────────
        if self.enable_mean_reversion:
            rsi_oversold:   float = float(self.cfg.get("rsi_oversold",   30))
            rsi_overbought: float = float(self.cfg.get("rsi_overbought", 70))

            bb_not_expanding: pd.Series = (
                df["bb_bandwidth"]
                < df["bb_bandwidth_sma"] * self.cfg["bb_bandwidth_expansion_factor"]
            )

            # LONG: oversold di tepi bawah BB — peluang reversal ke atas
            mean_rev_long: pd.Series = (
                (df["regime"] == REGIME_SIDEWAYS)
                & (df["close"] <= df["bb_lower"])
                & (df["rsi"] < rsi_oversold)
                & bb_not_expanding
                & volume_confirmed
            )
            # SHORT: overbought di tepi atas BB — peluang reversal ke bawah
            mean_rev_short: pd.Series = (
                (df["regime"] == REGIME_SIDEWAYS)
                & (df["close"] >= df["bb_upper"])
                & (df["rsi"] > rsi_overbought)
                & bb_not_expanding
                & volume_confirmed
            )
        else:
            mean_rev_long  = pd.Series(False, index=df.index)
            mean_rev_short = pd.Series(False, index=df.index)

        # ──────────────────────────────────────────────────────────────────
        # Gabungkan sinyal — np.select (first-match; no overlap by design)
        #   Overlap tidak mungkin karena:
        #   • EMA/Supertrend direction adalah mutually exclusive
        #   • bb_lower < bb_upper secara definisi
        # ──────────────────────────────────────────────────────────────────
        conditions = [
            trend_long,           trend_short,
            mean_rev_long,        mean_rev_short,
        ]
        signal_values   = [SIGNAL_BUY,            SIGNAL_SELL,
                           SIGNAL_BUY,            SIGNAL_SELL]
        strategy_labels = [STRATEGY_TREND_LONG,   STRATEGY_TREND_SHORT,
                           STRATEGY_MEAN_REV_LONG, STRATEGY_MEAN_REV_SHORT]

        df["signal"]       = np.select(conditions, signal_values,   default=SIGNAL_HOLD)
        df["strategy_used"] = np.select(conditions, strategy_labels, default=STRATEGY_NONE)

        # ── Logging ringkasan ─────────────────────────────────────────────
        n_tl = int(trend_long.sum())
        n_ts = int(trend_short.sum())
        n_ml = int(mean_rev_long.sum())
        n_ms = int(mean_rev_short.sum())

        logger.info(
            "[SignalGenerator] Sinyal dihasilkan | "
            "TrendLong: %d | TrendShort: %d | "
            "MeanRevLong: %d | MeanRevShort: %d | Total: %d",
            n_tl, n_ts, n_ml, n_ms,
            n_tl + n_ts + n_ml + n_ms,
        )

        return df

    # ------------------------------------------------------------------
    # Validasi Kolom (fail-fast)
    # ------------------------------------------------------------------
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Memastikan semua kolom yang diperlukan tersedia di DataFrame.

        Args:
            df: DataFrame yang akan divalidasi.

        Raises:
            KeyError: Jika ada kolom yang tidak ditemukan.
        """
        required = [
            "regime", "ema_fast", "ema_slow",
            "macd_histogram", "supertrend_dir",
            "close", "bb_lower", "bb_upper", "bb_bandwidth", "bb_bandwidth_sma",
            "rsi", "volume", "volume_sma",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Kolom berikut tidak ditemukan di DataFrame: {missing}. "
                "Pastikan IndicatorCalculator.calculate_all() dan "
                "RegimeFilter.classify() sudah dijalankan."
            )
