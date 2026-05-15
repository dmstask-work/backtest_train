"""
Mesin Backtesting Utama (Fase 5) — v3 Institutional Grade.

Refactor menyeluruh untuk menghilangkan tiga bias struktural yang membuat
hasil backtest tidak dapat dipercaya untuk deployment produksi:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  BIAS YANG DIHILANGKAN                                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │  1. LOOKAHEAD BIAS (Entry)                                          │
  │     Dulu : Signal di close[i] → entry di close[i] (mustahil nyata) │
  │     Kini  : Signal di close[i] → pending order → entry di open[i+1]│
  │                                                                     │
  │  2. OPTIMISTIC INTRABAR EXIT                                        │
  │     Dulu : TP menang jika SL & TP keduanya kena dalam 1 candle     │
  │     Kini  : SL menang (pessimistic worst-case assumption)           │
  │                                                                     │
  │  3. ZERO SLIPPAGE                                                   │
  │     Dulu : Eksekusi tepat di harga teori (tidak realistis)          │
  │     Kini  : Entry degraded +slippage_rate, Exit degraded -slippage  │
  └─────────────────────────────────────────────────────────────────────┘

Urutan Pemrosesan Per Candle (sesuai urutan real OHLC):
  [OPEN]     → Eksekusi pending order (entry) di open * (1 + slippage)
  [HIGH/LOW] → Evaluasi SL/TP: pessimistic (SL menang jika keduanya hit)
  [CLOSE]    → Update ekuitas MTM | Register pending order baru

Rumus Modal Bersih:
  equity = modal_bebas + (quantity × close)
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
    Simulator eksekusi trading institutional-grade dengan tiga perbaikan utama:

      • Pending Order System  — entry di open[i+1], bukan close[i]
      • Pessimistic Exit      — SL selalu menang atas TP jika terjadi bersamaan
      • Slippage Modeling     — harga eksekusi entry & exit didegradasi

    Pemisahan tanggung jawab (separation of concerns) dijaga penuh:
    SignalGenerator dan RiskManager tidak diubah, hanya cara
    BacktestEngine mengkonsumsi outputnya yang berubah.
    """

    def __init__(
        self,
        initial_capital:  float            = BACKTEST_CONFIG["initial_capital"],
        trade_allocation: float            = BACKTEST_CONFIG["trade_allocation"],
        fee_rate:         float            = BACKTEST_CONFIG["fee_rate"],
        slippage_rate:    float            = 0.0005,
        risk_manager:     Optional[RiskManager] = None,
    ) -> None:
        """
        Inisialisasi mesin backtesting v3.

        Args:
            initial_capital:  Modal awal simulasi (USD).
            trade_allocation: Fraksi modal yang dipakai per trade (0.0–1.0).
            fee_rate:         Biaya transaksi per sisi (0.001 = 0.1%).
            slippage_rate:    Degradasi harga akibat slippage per sisi
                              (0.0005 = 0.05%). Entry: harga naik sebesar ini.
                              Exit: harga turun sebesar ini.
            risk_manager:     Instance RiskManager. Dibuat otomatis jika None.
        """
        self.initial_capital  = initial_capital
        self.trade_allocation = trade_allocation
        self.fee_rate         = fee_rate
        self.slippage_rate    = slippage_rate
        self.risk_manager     = risk_manager or RiskManager()

        # State internal — di-reset di setiap pemanggilan run()
        self._capital:       float               = initial_capital
        self._trades:        List[Dict[str, Any]] = []
        self._equity_curve:  List[float]          = []
        self._in_position:   bool                 = False
        self._position:      Dict[str, Any]       = {}
        self._pending_order: Dict[str, Any]       = {}  # {} = tidak ada pending

    # ==================================================================
    # API Publik
    # ==================================================================
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Menjalankan simulasi backtest pada seluruh dataset.

        Urutan pemrosesan per candle mencerminkan urutan OHLC nyata:
          OPEN  → eksekusi pending entry dengan slippage
          H/L   → evaluasi SL/TP (pessimistic: SL menang jika ambig)
          CLOSE → update MTM equity | catat pending order baru

        Args:
            df: DataFrame lengkap dengan kolom sinyal, indikator, dan regime.
                Harus memiliki kolom: open, high, low, close, atr,
                signal, strategy_used, regime.

        Returns:
            Dictionary berisi:
              trades        (list):  Semua catatan trade yang terjadi.
              equity_curve  (list):  Nilai ekuitas mark-to-market per candle.
              final_capital (float): Modal akhir setelah semua posisi ditutup.
        """
        self._reset_state()

        logger.info(
            "[BacktestEngine] Mulai simulasi | %d candle | "
            "Modal: $%.2f | Alokasi: %.0f%% | Fee: %.2f%% | Slippage: %.2f%%",
            len(df),
            self.initial_capital,
            self.trade_allocation * 100,
            self.fee_rate * 100,
            self.slippage_rate * 100,
        )

        for row in df.itertuples():
            idx = row.Index

            # ── [OPEN] Eksekusi pending order di harga open + slippage ─
            # Ini menghilangkan lookahead bias: signal candle sebelumnya
            # baru dieksekusi sekarang di harga open candle ini.
            if self._pending_order and not self._in_position:
                self._execute_pending_entry(idx, row)

            # ── [HIGH/LOW] Evaluasi exit dengan resolusi pessimistik ───
            # Urutan: cek candle ini dulu sebelum MTM, agar posisi yang
            # baru saja dibuka bisa langsung di-stop jika open = gap down.
            if self._in_position:
                self._check_exit(idx, row)

            # ── [CLOSE] Update ekuitas mark-to-market ─────────────────
            if self._in_position:
                mtm = self._capital + (
                    self._position["quantity"] * float(row.close)
                )
            else:
                mtm = self._capital
            self._equity_curve.append(mtm)

            # ── [CLOSE] Register pending order untuk candle berikutnya ─
            # HANYA jika tidak sedang dalam posisi DAN belum ada pending.
            # Signal yang diterima sekarang → dieksekusi di open[i+1].
            if (
                not self._in_position
                and not self._pending_order
                and int(row.signal) == SIGNAL_BUY
            ):
                self._pending_order = {
                    "signal_time": idx,
                    "atr":         float(row.atr),
                    "strategy":    str(row.strategy_used),
                    "regime":      str(row.regime),
                }
                logger.debug(
                    "[PENDING] %s | Signal diterima, menunggu open candle berikutnya.",
                    idx,
                )

        # ── Tutup paksa posisi yang masih terbuka di akhir data ───────
        # Slippage diterapkan: close * (1 - slippage_rate)
        if self._in_position:
            last       = df.iloc[-1]
            eod_price  = float(last["close"]) * (1.0 - self.slippage_rate)
            logger.info(
                "[BacktestEngine] Menutup posisi aktif di akhir data | "
                "Close: $%.4f | Exec (after slippage): $%.4f",
                float(last["close"]),
                eod_price,
            )
            self._close_position(
                idx        = df.index[-1],
                exec_price = eod_price,
                exit_reason= "END_OF_DATA",
            )

        # Pending order yang tidak sempat dieksekusi dibuang (tidak ada data lagi)
        if self._pending_order:
            logger.debug(
                "[BacktestEngine] Pending order pada %s dibuang (akhir data).",
                self._pending_order.get("signal_time"),
            )
            self._pending_order = {}

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
        self._capital       = self.initial_capital
        self._trades        = []
        self._equity_curve  = []
        self._in_position   = False
        self._position      = {}
        self._pending_order = {}

    def _execute_pending_entry(self, idx: Any, row: Any) -> None:
        """
        Mengeksekusi pending order di harga open candle saat ini + slippage.

        Dipanggil di fase [OPEN] — ini adalah perbaikan utama anti-lookahead.
        Harga entry bukan lagi close[i] tapi open[i+1] yang sudah terdegradasi
        oleh slippage_rate untuk mensimulasikan kondisi pasar nyata.

        SL dan TP dihitung berdasarkan harga eksekusi aktual (bukan teori),
        dengan ATR yang diambil dari candle sinyal (candle sebelumnya).

        Args:
            idx: Timestamp candle eksekusi (open[i+1]).
            row: Named tuple baris DataFrame candle saat ini.
        """
        raw_open:   float = float(row.open)
        exec_price: float = raw_open * (1.0 + self.slippage_rate)

        # SL/TP dikalkulasi dari exec_price (bukan raw open)
        levels = self.risk_manager.calculate_levels(
            entry_price = exec_price,
            atr         = self._pending_order["atr"],
        )
        sizing = self.risk_manager.calculate_position_size(
            capital        = self._capital,
            allocation_pct = self.trade_allocation,
            entry_price    = exec_price,
            fee_rate       = self.fee_rate,
        )

        self._in_position = True
        self._position = {
            "signal_time":     self._pending_order["signal_time"],
            "entry_time":      idx,
            "raw_entry_price": raw_open,
            "entry_price":     exec_price,          # harga eksekusi nyata
            "slippage_entry":  exec_price - raw_open,
            "quantity":        sizing["quantity"],
            "trade_capital":   sizing["trade_capital"],
            "entry_fee":       sizing["entry_fee"],
            "stop_loss":       levels["stop_loss"],
            "take_profit":     levels["take_profit"],
            "risk_reward":     levels["risk_reward_ratio"],
            "strategy":        self._pending_order["strategy"],
            "regime":          self._pending_order["regime"],
            "atr":             self._pending_order["atr"],
        }

        self._capital      -= sizing["trade_capital"]
        self._pending_order = {}   # Hapus pending order yang sudah dieksekusi

        logger.debug(
            "[ENTRY] %s | Open: $%.4f → Exec: $%.4f (+slip $%.4f) | "
            "SL: $%.4f | TP: $%.4f | %s",
            idx,
            raw_open,
            exec_price,
            exec_price - raw_open,
            levels["stop_loss"],
            levels["take_profit"],
            self._position["strategy"],
        )

    def _check_exit(self, idx: Any, row: Any) -> None:
        """
        Mengevaluasi kondisi exit dengan resolusi intrabar pessimistik.

        PERUBAHAN KRITIS dari v2:
          Jika high[i] >= TP DAN low[i] <= SL dalam candle yang sama,
          sebelumnya TP dianggap menang (optimistic). Kini SL selalu
          diprioritas (pessimistic worst-case), karena dalam kondisi
          volatilitas tinggi harga sangat mungkin menembus SL lebih dulu.

        Slippage diterapkan ke harga eksekusi exit:
          SL exit: exec = sl_price  * (1 - slippage_rate)  [lebih buruk]
          TP exit: exec = tp_price  * (1 - slippage_rate)  [sedikit lebih buruk]

        Args:
            idx: Timestamp candle saat ini.
            row: Named tuple baris DataFrame.
        """
        hit_tp: bool = float(row.high) >= self._position["take_profit"]
        hit_sl: bool = float(row.low)  <= self._position["stop_loss"]

        if hit_sl and hit_tp:
            # ── PESSIMISTIC: kedua level tersentuh dalam satu candle ──
            # Asumsi: SL selalu kena lebih dulu (worst-case institutional)
            logger.debug(
                "[EXIT] %s | ⚠ SL & TP keduanya hit — resolusi PESSIMISTIC → SL",
                idx,
            )
            exec_price = self._position["stop_loss"] * (1.0 - self.slippage_rate)
            self._close_position(idx, exec_price, "STOP_LOSS")

        elif hit_sl:
            exec_price = self._position["stop_loss"] * (1.0 - self.slippage_rate)
            self._close_position(idx, exec_price, "STOP_LOSS")

        elif hit_tp:
            exec_price = self._position["take_profit"] * (1.0 - self.slippage_rate)
            self._close_position(idx, exec_price, "TAKE_PROFIT")

    def _close_position(
        self,
        idx:         Any,
        exec_price:  float,
        exit_reason: str,
    ) -> None:
        """
        Menutup posisi aktif, menghitung PnL net, dan mencatat ke trade log.

        PnL dihitung dari harga eksekusi aktual (sudah termasuk slippage),
        bukan dari harga teori SL/TP. Fee exit dihitung dari nilai eksekusi.

        Args:
            idx:         Timestamp penutupan posisi.
            exec_price:  Harga eksekusi exit aktual (sudah terdegradasi slippage).
            exit_reason: 'TAKE_PROFIT', 'STOP_LOSS', atau 'END_OF_DATA'.
        """
        pos = self._position

        exit_fee:   float = (pos["quantity"] * exec_price) * self.fee_rate
        gross_pnl:  float = (exec_price - pos["entry_price"]) * pos["quantity"]
        total_fees: float = pos["entry_fee"] + exit_fee
        net_pnl:    float = gross_pnl - total_fees

        # Kembalikan modal: saldo bebas + trade_capital + net_pnl
        self._capital += pos["trade_capital"] + net_pnl

        trade_record: Dict[str, Any] = {
            # ── Waktu ────────────────────────────────────────────────
            "signal_time":       pos["signal_time"],
            "entry_time":        pos["entry_time"],
            "exit_time":         idx,
            # ── Harga ────────────────────────────────────────────────
            "raw_entry_price":   pos["raw_entry_price"],
            "entry_price":       pos["entry_price"],      # after slippage
            "exit_price":        exec_price,              # after slippage
            "stop_loss":         pos["stop_loss"],
            "take_profit":       pos["take_profit"],
            # ── Posisi ───────────────────────────────────────────────
            "quantity":          pos["quantity"],
            "strategy":          pos["strategy"],
            "regime":            pos["regime"],
            "risk_reward":       pos["risk_reward"],
            # ── PnL ──────────────────────────────────────────────────
            "exit_reason":       exit_reason,
            "gross_pnl":         gross_pnl,
            "total_fees":        total_fees,
            "net_pnl":           net_pnl,
            "outcome":           "WIN" if net_pnl > 0 else "LOSS",
            # ── Diagnostik ───────────────────────────────────────────
            "slippage_entry":    pos["slippage_entry"],
            "slippage_exit":     abs(exec_price - (
                pos["stop_loss"] if exit_reason == "STOP_LOSS" else
                pos["take_profit"] if exit_reason == "TAKE_PROFIT" else
                exec_price / (1.0 - self.slippage_rate) * self.slippage_rate
            )),
        }

        self._trades.append(trade_record)
        self._in_position = False
        self._position    = {}

        logger.debug(
            "[EXIT] %s | %s | Exec: $%.4f | Net PnL: $%+.4f | %s",
            idx,
            exit_reason,
            exec_price,
            net_pnl,
            trade_record["outcome"],
        )


