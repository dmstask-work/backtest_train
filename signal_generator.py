"""
Modul Generator Sinyal Trading Adaptif (Fase 3) — v2.

Menghasilkan sinyal entry BUY berdasarkan kondisi pasar (regime) saat ini.
Seluruh logika diterapkan secara vectorized sehingga tidak ada loop eksplisit.

Strategi yang Digunakan:
─────────────────────────────────────────────────────────────────────────
  Kondisi A │ regime = TRENDING  → Trend-Following
             │  • EMA fast > EMA slow          [tren naik aktif]
             │  • MACD histogram cross-up 0    [momentum konfirmasi]
             │  • Harga Close > Supertrend     [filter anti-whipsaw]   ← BARU
             │  • Volume > Volume SMA          [konfirmasi volume]     ← BARU
─────────────────────────────────────────────────────────────────────────
  Kondisi B │ regime = SIDEWAYS  → Mean-Reversion
             │  • Close ≤ Lower Bollinger Band [harga di tepi bawah]
             │  • RSI < rsi_oversold           [kondisi oversold]
             │  • BB Bandwidth < SMA × factor  [BB tidak expanding]   ← BARU
             │  • Volume > Volume SMA          [konfirmasi volume]     ← BARU
─────────────────────────────────────────────────────────────────────────

Strategy Switcher:
  enable_trend_following dan enable_mean_reversion dapat dimatikan
  secara independen melalui CLI atau settings.json untuk A/B testing.
  Kode strategi yang dimatikan TIDAK dihapus — hanya di-skip saat
  evaluasi sinyal.
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
      - Pasar trending  → Trend-Following (EMA + MACD + Supertrend + Volume)
      - Pasar sideways  → Mean-Reversion  (BB + RSI + Bandwidth filter + Volume)

    Setiap strategi dapat dimatikan secara independen via saklar
    enable_trend_following / enable_mean_reversion tanpa menghapus kodenya,
    sehingga mudah untuk A/B testing dan debugging.
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
            enable_trend_following:  Aktifkan strategi Trend-Following.
                                     Default: True.
            enable_mean_reversion:   Aktifkan strategi Mean-Reversion.
                                     Default: True.
        """
        self.cfg = config
        self.enable_trend_following = enable_trend_following
        self.enable_mean_reversion  = enable_mean_reversion

        if not enable_trend_following and not enable_mean_reversion:
            logger.warning(
                "[SignalGenerator] ⚠ Kedua strategi dinonaktifkan! "
                "Tidak ada sinyal BUY yang akan dihasilkan."
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
              signal        (int): 1 = BUY, 0 = HOLD
              strategy_used (str): Label strategi yang memicu sinyal
        """
        self._validate_columns(df)
        df = df.copy()

        # ==============================================================
        # FILTER UNIVERSAL — Konfirmasi Volume (berlaku untuk semua sinyal)
        # Tolak sinyal jika volume candle ini < rata-rata volume (SMA Vol)
        # Tujuan: menyaring fake breakout yang digerakkan volume rendah
        # ==============================================================
        volume_confirmed: pd.Series = df["volume"] > df["volume_sma"]

        # ==============================================================
        # KONDISI A — Trend-Following (aktif saat TRENDING)
        # ==============================================================
        # Logika:
        #   1. EMA fast > EMA slow          → struktur tren naik terkonfirmasi
        #   2. MACD histogram cross-up 0    → momentum baru mulai naik
        #   3. Supertrend direction = +1    → harga masih di atas supertrend
        #      (mencegah entry saat tren palsu / whipsaw)
        #   4. Volume konfirmasi            → breakout didukung volume nyata
        # ==============================================================
        if self.enable_trend_following:
            macd_cross_up: pd.Series = (
                (df["macd_histogram"] > 0)
                & (df["macd_histogram"].shift(1) <= 0)
            )
            trend_buy: pd.Series = (
                (df["regime"] == REGIME_TRENDING)
                & (df["ema_fast"] > df["ema_slow"])
                & macd_cross_up
                & (df["supertrend_dir"] == 1)          # filter anti-whipsaw
                & volume_confirmed                      # filter volume
            )
        else:
            # Strategi dimatikan — semua False (tidak ada sinyal)
            trend_buy = pd.Series(False, index=df.index)

        # ==============================================================
        # KONDISI B — Mean-Reversion (aktif saat SIDEWAYS)
        # ==============================================================
        # Logika:
        #   1. Close <= BB Lower            → harga menyentuh tepi bawah
        #   2. RSI < rsi_oversold (30)      → kondisi oversold terkonfirmasi
        #   3. BB Bandwidth < SMA × factor  → BB tidak sedang expanding tajam
        #      (mencegah entry saat pasar sedang dump/crash — "pisau jatuh")
        #   4. Volume konfirmasi            → reversal didukung volume
        # ==============================================================
        if self.enable_mean_reversion:
            bb_not_expanding: pd.Series = (
                df["bb_bandwidth"]
                < df["bb_bandwidth_sma"] * self.cfg["bb_bandwidth_expansion_factor"]
            )
            mean_rev_buy: pd.Series = (
                (df["regime"] == REGIME_SIDEWAYS)
                & (df["close"] <= df["bb_lower"])
                & (df["rsi"] < self.cfg["rsi_oversold"])
                & bb_not_expanding                      # filter anti-dump
                & volume_confirmed                      # filter volume
            )
        else:
            # Strategi dimatikan — semua False
            mean_rev_buy = pd.Series(False, index=df.index)

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
        n_trend = int(trend_buy.sum())
        n_mean  = int(mean_rev_buy.sum())
        n_total = n_trend + n_mean

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
            "macd_histogram", "supertrend_dir",
            "close", "bb_lower", "bb_bandwidth", "bb_bandwidth_sma",
            "rsi", "volume", "volume_sma",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Kolom berikut tidak ditemukan di DataFrame: {missing}. "
                "Pastikan IndicatorCalculator.calculate_all() dan "
                "RegimeFilter.classify() sudah dijalankan."
            )
