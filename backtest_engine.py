"""
Mesin Backtesting Utama (Fase 5).

Mensimulasikan eksekusi trading satu posisi long pada satu waktu,
mengelola entry/exit berdasarkan sinyal dan level SL/TP.

Desain Mesin:
  • Single-position mode — hanya boleh ada satu posisi aktif.
  • Sequential execution — loop itertuples (lebih cepat dari iterrows).
  • SL/TP diperiksa menggunakan candle high/low (worst-case simulation).
  • Jika SL dan TP keduanya terkena dalam satu candle → TP dianggap lebih
    dulu tercapai (optimistic assumption; bisa diubah jika perlu).
  • Posisi yang masih terbuka di akhir data akan ditutup pada harga close.

Rumus Modal Bersih:
  equity = modal_bebas + (quantity × harga_sekarang)
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from config import BACKTEST_CONFIG
from risk_manager import RiskManager
from signal_generator import SIGNAL_BUY

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Simulasi eksekusi trading berdasarkan sinyal adaptif.

    Mensimulasikan entry berdasarkan sinyal, mengelola SL/TP, dan mencatat
    setiap trade beserta hasil PnL-nya. Semua biaya transaksi (fee) turut
    diperhitungkan pada sisi entry dan exit.
    """

    def __init__(
        self,
        initial_capital: float     = BACKTEST_CONFIG["initial_capital"],
        trade_allocation: float    = BACKTEST_CONFIG["trade_allocation"],
        fee_rate: float            = BACKTEST_CONFIG["fee_rate"],
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        """
        Inisialisasi mesin backtesting.

        Args:
            initial_capital:  Modal awal simulasi (USD).
            trade_allocation: Fraksi modal yang dipakai per trade (0.0–1.0).
            fee_rate:         Biaya transaksi per sisi (0.001 = 0.1%).
            risk_manager:     Instance RiskManager. Dibuat otomatis jika None.
        """
        self.initial_capital  = initial_capital
        self.trade_allocation = trade_allocation
        self.fee_rate         = fee_rate
        self.risk_manager     = risk_manager or RiskManager()

        # State internal (di-reset setiap run)
        self._capital: float                  = initial_capital
        self._trades: List[Dict[str, Any]]    = []
        self._equity_curve: List[float]       = []
        self._in_position: bool               = False
        self._position: Dict[str, Any]        = {}

    # ==================================================================
    # API Publik
    # ==================================================================
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Menjalankan simulasi backtest pada seluruh dataset.

        Args:
            df: DataFrame lengkap dengan kolom sinyal, indikator, dan regime.

        Returns:
            Dictionary berisi:
              trades       (list):  Semua catatan trade yang terjadi.
              equity_curve (list):  Nilai ekuitas mark-to-market per candle.
              final_capital (float): Modal akhir setelah semua posisi ditutup.
        """
        self._reset_state()

        logger.info(
            "[BacktestEngine] Mulai simulasi | %d candle | "
            "Modal: $%.2f | Alokasi: %.0f%% | Fee: %.2f%%",
            len(df),
            self.initial_capital,
            self.trade_allocation * 100,
            self.fee_rate * 100,
        )

        # Loop sequential menggunakan itertuples (lebih cepat dari iterrows)
        for row in df.itertuples():
            idx           = row.Index
            current_price = row.close

            # ---- 1. Update kurva ekuitas mark-to-market ----
            if self._in_position:
                mtm = self._capital + (self._position["quantity"] * current_price)
            else:
                mtm = self._capital
            self._equity_curve.append(mtm)

            # ---- 2. Cek kondisi exit jika ada posisi aktif ----
            if self._in_position:
                self._check_exit(idx, row)

            # ---- 3. Cek kondisi entry jika tidak ada posisi aktif ----
            if not self._in_position and int(row.signal) == SIGNAL_BUY:
                self._execute_entry(idx, row)

        # ---- Tutup paksa posisi yang masih terbuka di akhir data ----
        if self._in_position:
            last = df.iloc[-1]
            logger.info(
                "[BacktestEngine] Menutup posisi aktif di akhir data | "
                "Harga close: $%.4f",
                float(last["close"]),
            )
            self._close_position(
                idx=df.index[-1],
                exit_price=float(last["close"]),
                exit_reason="END_OF_DATA",
            )

        logger.info(
            "[BacktestEngine] Simulasi selesai | Total trade: %d | "
            "Modal akhir: $%.4f",
            len(self._trades),
            self._capital,
        )

        return {
            "trades":        self._trades,
            "equity_curve":  self._equity_curve,
            "final_capital": self._capital,
        }

    # ==================================================================
    # Metode Internal
    # ==================================================================
    def _reset_state(self) -> None:
        """Mereset semua state engine ke nilai awal sebelum setiap run."""
        self._capital      = self.initial_capital
        self._trades       = []
        self._equity_curve = []
        self._in_position  = False
        self._position     = {}

    def _execute_entry(self, idx: Any, row: Any) -> None:
        """
        Mengeksekusi entry posisi long baru.

        Args:
            idx: Timestamp (index) candle entry.
            row: Named tuple baris DataFrame (dari itertuples).
        """
        levels = self.risk_manager.calculate_levels(
            entry_price=float(row.close),
            atr=float(row.atr),
        )
        sizing = self.risk_manager.calculate_position_size(
            capital=self._capital,
            allocation_pct=self.trade_allocation,
            entry_price=float(row.close),
            fee_rate=self.fee_rate,
        )

        self._in_position = True
        self._position = {
            "entry_time":    idx,
            "entry_price":   float(row.close),
            "quantity":      sizing["quantity"],
            "trade_capital": sizing["trade_capital"],
            "entry_fee":     sizing["entry_fee"],
            "stop_loss":     levels["stop_loss"],
            "take_profit":   levels["take_profit"],
            "risk_reward":   levels["risk_reward_ratio"],
            "strategy":      str(row.strategy_used),
            "regime":        str(row.regime),
            "atr":           float(row.atr),
        }

        # Kurangi modal bebas sebesar trade_capital
        self._capital -= sizing["trade_capital"]

        logger.debug(
            "[ENTRY] %s | Close: $%.4f | SL: $%.4f | TP: $%.4f | %s",
            idx,
            float(row.close),
            levels["stop_loss"],
            levels["take_profit"],
            str(row.strategy_used),
        )

    def _check_exit(self, idx: Any, row: Any) -> None:
        """
        Memeriksa apakah SL atau TP tercapai pada candle ini.

        Menggunakan high dan low candle sebagai proxy harga ekstrem.
        Jika keduanya terpenuhi dalam satu candle → TP diprioritaskan.

        Args:
            idx: Timestamp (index) candle saat ini.
            row: Named tuple baris DataFrame.
        """
        hit_tp = float(row.high) >= self._position["take_profit"]
        hit_sl = float(row.low)  <= self._position["stop_loss"]

        if hit_tp:
            # TP diprioritaskan (optimistic; ubah jika ingin realistic mode)
            self._close_position(idx, self._position["take_profit"], "TAKE_PROFIT")
        elif hit_sl:
            self._close_position(idx, self._position["stop_loss"], "STOP_LOSS")

    def _close_position(
        self,
        idx: Any,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """
        Menutup posisi aktif dan mencatat hasil trade ke log.

        Args:
            idx:         Timestamp penutupan.
            exit_price:  Harga penutupan posisi.
            exit_reason: 'TAKE_PROFIT', 'STOP_LOSS', atau 'END_OF_DATA'.
        """
        pos = self._position

        # Fee exit dihitung dari nilai posisi saat penutupan
        exit_fee   = (pos["quantity"] * exit_price) * self.fee_rate
        gross_pnl  = (exit_price - pos["entry_price"]) * pos["quantity"]
        total_fees = pos["entry_fee"] + exit_fee
        net_pnl    = gross_pnl - total_fees

        # Kembalikan modal: modal_bebas + trade_capital + net_pnl
        self._capital += pos["trade_capital"] + net_pnl

        trade_record: Dict[str, Any] = {
            "entry_time":   pos["entry_time"],
            "exit_time":    idx,
            "entry_price":  pos["entry_price"],
            "exit_price":   exit_price,
            "quantity":     pos["quantity"],
            "strategy":     pos["strategy"],
            "regime":       pos["regime"],
            "stop_loss":    pos["stop_loss"],
            "take_profit":  pos["take_profit"],
            "risk_reward":  pos["risk_reward"],
            "exit_reason":  exit_reason,
            "gross_pnl":    gross_pnl,
            "total_fees":   total_fees,
            "net_pnl":      net_pnl,
            "outcome":      "WIN" if net_pnl > 0 else "LOSS",
        }

        self._trades.append(trade_record)
        self._in_position = False
        self._position    = {}

        logger.debug(
            "[EXIT] %s | %s | Price: $%.4f | Net PnL: $%+.4f | %s",
            idx,
            exit_reason,
            exit_price,
            net_pnl,
            trade_record["outcome"],
        )
