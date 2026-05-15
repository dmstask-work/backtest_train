"""
Modul Manajemen Risiko berbasis ATR (Fase 4).

Menghitung level Stop Loss dan Take Profit menggunakan Average True Range
sebagai satuan risiko dinamis yang menyesuaikan volatilitas pasar.

Rumus:
  LONG  → Stop Loss   = Entry − (sl_multiplier × ATR)
           Take Profit = Entry + (tp_multiplier × ATR)

  SHORT → Stop Loss   = Entry + (sl_multiplier × ATR)  ← SL di atas entry
           Take Profit = Entry − (tp_multiplier × ATR)  ← TP di bawah entry

Risk/Reward ratio default = 1 : 2  (SL=1.5×ATR, TP=3.0×ATR)
"""

import logging

from config import BACKTEST_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manajer risiko yang menghitung SL, TP, dan ukuran posisi secara dinamis.

    Dengan menggunakan ATR sebagai satuan, level SL/TP secara otomatis
    menyesuaikan diri terhadap volatilitas pasar saat ini — ketat di pasar
    tenang, lebih longgar di pasar volatil.
    """

    def __init__(
        self,
        sl_multiplier: float = RISK_CONFIG["sl_multiplier"],
        tp_multiplier: float = RISK_CONFIG["tp_multiplier"],
    ) -> None:
        """
        Inisialisasi manajer risiko.

        Args:
            sl_multiplier: Pengali ATR untuk Stop Loss.   Default: 1.5
            tp_multiplier: Pengali ATR untuk Take Profit. Default: 3.0
        """
        if sl_multiplier <= 0 or tp_multiplier <= 0:
            raise ValueError("Multiplier SL dan TP harus bernilai positif.")
        if tp_multiplier <= sl_multiplier:
            logger.warning(
                "[RiskManager] Risk/Reward < 1:1 (TP ≤ SL). "
                "Pertimbangkan untuk meningkatkan tp_multiplier."
            )

        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier

    # ------------------------------------------------------------------
    # Kalkulasi Level SL & TP
    # ------------------------------------------------------------------
    def calculate_levels(
        self,
        entry_price: float,
        atr:         float,
        direction:   str = "LONG",
    ) -> dict:
        """
        Menghitung level Stop Loss dan Take Profit untuk satu posisi.

        Level SL/TP bersifat direksional:
          LONG  → SL di bawah entry, TP di atas entry
          SHORT → SL di atas entry, TP di bawah entry (level terbalik)

        Args:
            entry_price: Harga eksekusi masuk posisi (sudah termasuk slippage).
            atr:         Nilai ATR dari candle sinyal.
            direction:   Arah posisi — 'LONG' atau 'SHORT'. Default: 'LONG'.

        Returns:
            Dictionary berisi:
              stop_loss         (float): Harga SL
              take_profit       (float): Harga TP
              risk_reward_ratio (float): Rasio reward dibagi risk
        """
        sl_distance: float = self.sl_multiplier * atr
        tp_distance: float = self.tp_multiplier * atr

        if direction == "LONG":
            stop_loss   = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_loss   = entry_price + sl_distance   # SL di atas entry
            take_profit = entry_price - tp_distance   # TP di bawah entry

        rr_ratio: float = tp_distance / sl_distance if sl_distance > 0 else 0.0

        return {
            "stop_loss":         stop_loss,
            "take_profit":       take_profit,
            "risk_reward_ratio": rr_ratio,
        }

    # ------------------------------------------------------------------
    # Kalkulasi Ukuran Posisi
    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        capital: float,
        allocation_pct: float,
        entry_price: float,
        fee_rate: float = BACKTEST_CONFIG["fee_rate"],
    ) -> dict:
        """
        Menghitung ukuran posisi berdasarkan persentase alokasi modal.

        Biaya transaksi diperhitungkan sebelum membeli aset sehingga
        modal yang diinvestasikan bersih dari fee masuk.

        Args:
            capital:        Modal tersedia saat ini (USD).
            allocation_pct: Fraksi modal yang dialokasikan (0.0 – 1.0).
            entry_price:    Harga entry per unit aset.
            fee_rate:       Biaya transaksi per sisi (contoh: 0.001 = 0.1%).

        Returns:
            Dictionary berisi:
              trade_capital (float): Total modal yang dialokasikan
              quantity      (float): Jumlah unit aset yang dibeli
              entry_fee     (float): Biaya fee masuk
        """
        trade_capital: float = capital * allocation_pct
        entry_fee:     float = trade_capital * fee_rate
        investable:    float = trade_capital - entry_fee   # modal bersih setelah fee
        quantity:      float = investable / entry_price

        return {
            "trade_capital": trade_capital,
            "quantity":      quantity,
            "entry_fee":     entry_fee,
        }
