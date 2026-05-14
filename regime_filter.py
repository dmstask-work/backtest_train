"""
Modul Market Regime Filter (Fase 2).

Mendeteksi kondisi pasar (TRENDING vs SIDEWAYS) berdasarkan nilai ADX.
Klasifikasi dilakukan secara vectorized menggunakan numpy.where,
sehingga efisien untuk dataset berukuran besar.

Aturan:
  ADX > threshold  → regime = 'TRENDING'  (pasar bergerak tren kuat)
  ADX ≤ threshold  → regime = 'SIDEWAYS'  (pasar konsolidasi/ranging)
"""

import logging

import numpy as np
import pandas as pd

from config import INDICATOR_CONFIG

logger = logging.getLogger(__name__)

# Konstanta label regime — digunakan sebagai referensi di modul lain
REGIME_TRENDING: str = "TRENDING"
REGIME_SIDEWAYS: str = "SIDEWAYS"


class RegimeFilter:
    """
    Mengklasifikasikan setiap candle ke dalam kondisi pasar menggunakan ADX.

    Mengapa ADX?
        ADX (Average Directional Index) mengukur kekuatan tren tanpa
        memperhatikan arahnya. Nilai tinggi (>25) berarti pasar sedang
        bergerak dengan kuat (trending), nilai rendah (≤25) berarti pasar
        sedang ranging atau konsolidasi.
    """

    def __init__(
        self,
        adx_threshold: float = INDICATOR_CONFIG["adx_threshold"],
    ) -> None:
        """
        Inisialisasi filter regime.

        Args:
            adx_threshold: Nilai ambang ADX untuk memisahkan TRENDING dan
                           SIDEWAYS. Default diambil dari config.py (25).
        """
        self.adx_threshold = adx_threshold

    # ------------------------------------------------------------------
    # Fungsi Klasifikasi
    # ------------------------------------------------------------------
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Menambahkan kolom 'regime' ke DataFrame berdasarkan nilai ADX.

        Operasi dilakukan secara vectorized (numpy.where) — tidak ada loop.

        Args:
            df: DataFrame yang sudah memiliki kolom 'adx'.

        Returns:
            DataFrame dengan kolom tambahan:
              regime (str): 'TRENDING' atau 'SIDEWAYS' untuk setiap baris.

        Raises:
            ValueError: Jika kolom 'adx' tidak ditemukan di DataFrame.
        """
        if "adx" not in df.columns:
            raise ValueError(
                "Kolom 'adx' tidak ditemukan. "
                "Pastikan IndicatorCalculator.calculate_all() dipanggil lebih dulu."
            )

        df = df.copy()

        # Klasifikasi vectorized — O(n) tanpa Python-level loop
        df["regime"] = np.where(
            df["adx"] > self.adx_threshold,
            REGIME_TRENDING,
            REGIME_SIDEWAYS,
        )

        # Logging statistik distribusi regime
        n_total     = len(df)
        n_trending  = (df["regime"] == REGIME_TRENDING).sum()
        n_sideways  = (df["regime"] == REGIME_SIDEWAYS).sum()

        logger.info(
            "[RegimeFilter] Klasifikasi selesai (ADX threshold=%.0f) | "
            "TRENDING: %d (%.1f%%) | SIDEWAYS: %d (%.1f%%)",
            self.adx_threshold,
            n_trending, (n_trending / n_total) * 100,
            n_sideways, (n_sideways / n_total) * 100,
        )

        return df
