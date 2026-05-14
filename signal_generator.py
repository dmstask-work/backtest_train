"""
Modul Generator Sinyal Trading Adaptif (Fase 3).

Menghasilkan sinyal entry BUY berdasarkan kondisi pasar (regime) saat ini.
Seluruh logika diterapkan secara vectorized sehingga tidak ada loop eksplisit.

Strategi yang Digunakan:
─────────────────────────────────────────────────────────────────────────
  Kondisi A │ regime = TRENDING  → Trend-Following
             │  • EMA_fast (20) > EMA_slow (50)    [tren naik aktif]
             │  • MACD histogram memotong ke atas 0 [momentum konfirmasi]
─────────────────────────────────────────────────────────────────────────
  Kondisi B │ regime = SIDEWAYS  → Mean-Reversion
             │  • Close ≤ Lower Bollinger Band      [harga menyentuh tepi bawah]
             │  • RSI < 30                          [kondisi oversold]
─────────────────────────────────────────────────────────────────────────
"""

import logging

import numpy as np
import pandas as pd

from config import INDICATOR_CONFIG
from regime_filter import REGIME_SIDEWAYS, REGIME_TRENDING

logger = logging.getLogger(__name__)

# Konstanta nilai sinyal
SIGNAL_BUY:  int = 1
SIGNAL_HOLD: int = 0

# Label strategi (digunakan untuk analitik breakdown)
STRATEGY_TREND:   str = "TREND_FOLLOWING"
STRATEGY_MEAN_REV: str = "MEAN_REVERSION"
STRATEGY_NONE:    str = "NONE"


class SignalGenerator:
    """
    Menghasilkan sinyal BUY adaptif berdasarkan regime pasar yang terdeteksi.

    Dua strategi bekerja secara eksklusif sesuai kondisi:
      - Pasar trending  → manfaatkan momentum tren (EMA + MACD)
      - Pasar sideways  → manfaatkan pembalikan arah dari area ekstrem (BB + RSI)
    """

    def __init__(self, config: dict = INDICATOR_CONFIG) -> None:
        """
        Inisialisasi generator sinyal.

        Args:
            config: Dictionary parameter indikator dari config.py.
        """
        self.cfg = config

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
              signal        (int): 1 = BUY, 0 = HOLD
              strategy_used (str): Strategi yang memicu sinyal tersebut
        """
        self._validate_columns(df)
        df = df.copy()

        # ==============================================================
        # KONDISI A — Trend-Following (hanya aktif saat TRENDING)
        # ==============================================================
        # MACD histogram cross-up: candle sebelumnya ≤ 0, sekarang > 0
        macd_cross_up: pd.Series = (
            (df["macd_histogram"] > 0)
            & (df["macd_histogram"].shift(1) <= 0)
        )

        trend_buy: pd.Series = (
            (df["regime"] == REGIME_TRENDING)
            & (df["ema_fast"] > df["ema_slow"])
            & macd_cross_up
        )

        # ==============================================================
        # KONDISI B — Mean-Reversion (hanya aktif saat SIDEWAYS)
        # ==============================================================
        mean_rev_buy: pd.Series = (
            (df["regime"] == REGIME_SIDEWAYS)
            & (df["close"] <= df["bb_lower"])
            & (df["rsi"] < self.cfg["rsi_oversold"])
        )

        # ==============================================================
        # Gabungkan sinyal — vectorized
        # ==============================================================
        df["signal"] = np.where(
            trend_buy | mean_rev_buy,
            SIGNAL_BUY,
            SIGNAL_HOLD,
        )

        # Label strategi untuk laporan analitik
        df["strategy_used"] = np.where(
            trend_buy,
            STRATEGY_TREND,
            np.where(mean_rev_buy, STRATEGY_MEAN_REV, STRATEGY_NONE),
        )

        # Logging ringkasan sinyal
        n_trend   = int(trend_buy.sum())
        n_mean    = int(mean_rev_buy.sum())
        n_total   = n_trend + n_mean

        logger.info(
            "[SignalGenerator] Sinyal BUY dihasilkan: %d total "
            "(Trend-Following: %d | Mean-Reversion: %d)",
            n_total,
            n_trend,
            n_mean,
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
            "macd_histogram", "close", "bb_lower", "rsi",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Kolom berikut tidak ditemukan di DataFrame: {missing}. "
                "Pastikan IndicatorCalculator dan RegimeFilter sudah dijalankan."
            )
