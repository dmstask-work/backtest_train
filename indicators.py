"""
Modul Kalkulasi Indikator Teknikal (Fase 2 & 3 - Pendukung).

Menghitung seluruh indikator yang diperlukan secara vectorized menggunakan
library pandas-ta. Tidak ada loop Python — semua operasi dikerjakan oleh
NumPy/C di bawah hood sehingga sangat cepat.

Indikator yang dihitung:
  - ADX (14)         → Deteksi regime pasar
  - EMA (20 & 50)    → Trend-following signal
  - MACD (12,26,9)   → Konfirmasi momentum
  - Bollinger Bands  → Mean-reversion signal
  - RSI (14)         → Konfirmasi oversold/overbought
  - ATR (14)         → Kalkulasi Stop Loss & Take Profit
"""

import logging

import pandas as pd
import pandas_ta as ta

from config import INDICATOR_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper aman untuk mengambil kolom pandas_ta berdasarkan prefix
# ---------------------------------------------------------------------------
def _pick_col(result_df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Mengambil kolom pertama dari DataFrame hasil pandas_ta berdasarkan prefix.

    pandas_ta menamai kolom secara dinamis (mis. 'ADX_14', 'BBL_20_2.0')
    sehingga helper ini diperlukan agar kode tidak bergantung pada format
    nama persis yang bisa berubah antar versi.

    Args:
        result_df: DataFrame kembalian dari fungsi pandas_ta.
        prefix:    Awalan nama kolom yang dicari, contoh: 'ADX_', 'BBL_'.

    Returns:
        Pandas Series kolom yang cocok.

    Raises:
        KeyError: Jika tidak ada kolom yang cocok dengan prefix.
    """
    matched = [c for c in result_df.columns if c.startswith(prefix)]
    if not matched:
        raise KeyError(
            f"Kolom dengan prefix '{prefix}' tidak ditemukan. "
            f"Kolom tersedia: {list(result_df.columns)}"
        )
    return result_df[matched[0]]


# ---------------------------------------------------------------------------
# Kelas Utama
# ---------------------------------------------------------------------------
class IndicatorCalculator:
    """
    Menghitung semua indikator teknikal yang diperlukan oleh sistem backtest.

    Seluruh kalkulasi bersifat vectorized — tidak ada iterasi baris satu
    per satu. Baris yang mengandung NaN (periode warm-up) dihapus di akhir.
    """

    def __init__(self, config: dict = INDICATOR_CONFIG) -> None:
        """
        Inisialisasi kalkulator dengan parameter konfigurasi.

        Args:
            config: Dictionary parameter indikator dari config.py.
        """
        self.cfg = config

    # ------------------------------------------------------------------
    # Kalkulasi Utama
    # ------------------------------------------------------------------
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghitung semua indikator teknikal sekaligus secara vectorized.

        Args:
            df: DataFrame OHLCV dengan kolom open, high, low, close, volume.

        Returns:
            DataFrame yang telah dilengkapi kolom-kolom indikator berikut:
              adx, ema_fast, ema_slow, macd_line, macd_signal_line,
              macd_histogram, bb_upper, bb_middle, bb_lower,
              rsi, atr
        """
        df = df.copy()

        logger.info("[Indicators] Menghitung indikator teknikal secara vectorized ...")

        # ---- ADX (Average Directional Index) -------------------------
        adx_res = ta.adx(
            df["high"],
            df["low"],
            df["close"],
            length=self.cfg["adx_period"],
        )
        df["adx"] = _pick_col(adx_res, "ADX_")

        # ---- EMA Cepat & Lambat --------------------------------------
        df["ema_fast"] = ta.ema(df["close"], length=self.cfg["ema_fast"])
        df["ema_slow"] = ta.ema(df["close"], length=self.cfg["ema_slow"])

        # ---- MACD ----------------------------------------------------
        macd_res = ta.macd(
            df["close"],
            fast=self.cfg["macd_fast"],
            slow=self.cfg["macd_slow"],
            signal=self.cfg["macd_signal"],
        )
        df["macd_line"]        = _pick_col(macd_res, "MACD_")
        df["macd_signal_line"] = _pick_col(macd_res, "MACDs_")
        df["macd_histogram"]   = _pick_col(macd_res, "MACDh_")

        # ---- Bollinger Bands -----------------------------------------
        bb_res = ta.bbands(
            df["close"],
            length=self.cfg["bb_period"],
            std=self.cfg["bb_std"],
        )
        df["bb_upper"]  = _pick_col(bb_res, "BBU_")
        df["bb_middle"] = _pick_col(bb_res, "BBM_")
        df["bb_lower"]  = _pick_col(bb_res, "BBL_")

        # ---- RSI (Relative Strength Index) ---------------------------
        df["rsi"] = ta.rsi(df["close"], length=self.cfg["rsi_period"])

        # ---- ATR (Average True Range) --------------------------------
        df["atr"] = ta.atr(
            df["high"],
            df["low"],
            df["close"],
            length=self.cfg["atr_period"],
        )

        # ---- Buang baris warm-up (NaN) -------------------------------
        rows_before = len(df)
        df.dropna(inplace=True)
        rows_dropped = rows_before - len(df)

        logger.info(
            "[Indicators] Selesai. %d baris warm-up dihapus. Data valid: %d baris.",
            rows_dropped,
            len(df),
        )

        return df
