"""
Modul Laporan Performa Backtesting (Fase 5 - Output) — v2.

Menghitung semua metrik performa dari hasil simulasi dan mencetak
laporan dalam dua format:
  1. Teks terstruktur yang mudah dibaca di terminal (CLI / VPS log)
  2. JSON kompak yang siap dikonsumsi oleh AI Agent / dashboard

Metrik yang Dihitung:
  ─ Ringkasan modal       : Modal awal, akhir, Net PnL, Total Return %
  ─ Statistik trade       : Total, Win, Loss, Win Rate, Avg Win/Loss, Profit Factor
  ─ Risiko                : Max Drawdown (%), Sharpe Ratio (per-candle, RF=0%)
  ─ Directional breakdown : LONG vs SHORT — Trades, Win Rate, Total PnL
  ─ Per sub-strategi      : Trades, Win Rate, PnL untuk semua label strategi
  ─ Breakdown exit        : Jumlah TP vs SL vs End-of-Data

Catatan Sharpe Ratio:
  Dihitung dari perubahan ekuitas antar-candle (kurva MTM).
  Formula: mean(returns) / std(returns), Risk-Free Rate = 0%.
  Tidak diannualisasi — timeframe tidak diketahui di level reporter.
"""

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """
    Menghitung metrik dan mencetak laporan hasil backtesting.

    Mendukung output ke terminal (teks rapi) dan JSON untuk integrasi
    dengan AI Agent atau sistem monitoring eksternal.
    Kompatibel penuh dengan engine v3 (Long/Short) dan engine lama (Long-only).
    """

    def __init__(self, initial_capital: float) -> None:
        """
        Inisialisasi reporter.

        Args:
            initial_capital: Modal awal yang digunakan dalam simulasi.
        """
        self.initial_capital = initial_capital

    # ==================================================================
    # Kalkulasi Metrik
    # ==================================================================
    def calculate_metrics(
        self,
        trades:        List[Dict[str, Any]],
        equity_curve:  List[float],
        final_capital: float,
    ) -> Dict[str, Any]:
        """
        Menghitung semua metrik performa dari hasil backtest.

        Args:
            trades:        Daftar catatan trade dari BacktestEngine.run().
            equity_curve:  Kurva ekuitas mark-to-market per candle.
            final_capital: Modal akhir setelah semua posisi ditutup.

        Returns:
            Dictionary terstruktur berisi semua metrik performa.
        """
        if not trades:
            logger.warning(
                "[Reporter] Tidak ada trade yang terjadi selama periode backtest."
            )
            return {
                "error": "Tidak ada trade yang dieksekusi selama periode backtest."
            }

        df_t = pd.DataFrame(trades)

        # ── Metrik Dasar ──────────────────────────────────────────────
        total_trades = len(df_t)
        n_win        = int((df_t["outcome"] == "WIN").sum())
        n_loss       = int((df_t["outcome"] == "LOSS").sum())
        win_rate     = (n_win / total_trades * 100) if total_trades > 0 else 0.0

        # ── PnL Agregat ───────────────────────────────────────────────
        total_net_pnl    = float(df_t["net_pnl"].sum())
        total_return_pct = (total_net_pnl / self.initial_capital) * 100
        total_fees       = float(df_t["total_fees"].sum())

        wins_df   = df_t[df_t["outcome"] == "WIN"]["net_pnl"]
        losses_df = df_t[df_t["outcome"] == "LOSS"]["net_pnl"]

        avg_win  = float(wins_df.mean())   if len(wins_df)   > 0 else 0.0
        avg_loss = float(losses_df.mean()) if len(losses_df) > 0 else 0.0

        # ── Profit Factor ─────────────────────────────────────────────
        gross_profit  = float(df_t[df_t["net_pnl"] > 0]["net_pnl"].sum())
        gross_loss    = abs(float(df_t[df_t["net_pnl"] < 0]["net_pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # ── Maximum Drawdown (Peak-to-Trough) ─────────────────────────
        equity_series = pd.Series(equity_curve, dtype=float)
        rolling_peak  = equity_series.expanding().max()
        drawdown_pct  = ((equity_series - rolling_peak) / rolling_peak) * 100
        max_drawdown  = float(drawdown_pct.min())   # nilai negatif

        # ── Sharpe Ratio (per-candle, Risk-Free Rate = 0%) ────────────
        # Formula: mean(Δequity%) / std(Δequity%)
        # Berbasis kurva ekuitas MTM per-candle; bukan diannualisasi
        # karena timeframe tidak diketahui di level reporter ini.
        returns     = equity_series.pct_change().dropna()
        returns_std = float(returns.std())
        sharpe_ratio = (
            float(returns.mean() / returns_std)
            if (len(returns) > 1 and returns_std > 0)
            else 0.0
        )

        # ── Directional Breakdown (LONG vs SHORT) ─────────────────────
        # Kompatibel mundur: jika engine < v3 tidak memproduksi
        # 'position_type', seluruh trade diasumsikan LONG.
        directional_breakdown: Dict[str, Any] = {}
        if "position_type" in df_t.columns:
            for direction in ("LONG", "SHORT"):
                ddf   = df_t[df_t["position_type"] == direction]
                if ddf.empty:
                    directional_breakdown[direction] = {
                        "total_trades":  0,
                        "wins":          0,
                        "losses":        0,
                        "win_rate_pct":  0.0,
                        "total_pnl_usd": 0.0,
                    }
                    continue
                d_win = int((ddf["outcome"] == "WIN").sum())
                directional_breakdown[direction] = {
                    "total_trades":  len(ddf),
                    "wins":          d_win,
                    "losses":        len(ddf) - d_win,
                    "win_rate_pct":  round((d_win / len(ddf)) * 100, 2),
                    "total_pnl_usd": round(float(ddf["net_pnl"].sum()), 4),
                }
        else:
            # Engine < v3: hanya ada posisi LONG
            directional_breakdown["LONG"] = {
                "total_trades":  total_trades,
                "wins":          n_win,
                "losses":        n_loss,
                "win_rate_pct":  round(win_rate, 2),
                "total_pnl_usd": round(total_net_pnl, 4),
                "note":          "position_type tidak tersedia (engine < v3)",
            }

        # ── Breakdown per Sub-Strategi ────────────────────────────────
        strategy_breakdown: Dict[str, Any] = {}
        for strat in df_t["strategy"].unique():
            sdf   = df_t[df_t["strategy"] == strat]
            s_win = int((sdf["outcome"] == "WIN").sum())
            strategy_breakdown[strat] = {
                "total_trades":  len(sdf),
                "wins":          s_win,
                "losses":        len(sdf) - s_win,
                "win_rate_pct":  round((s_win / len(sdf)) * 100, 2),
                "total_pnl_usd": round(float(sdf["net_pnl"].sum()), 4),
            }

        # ── Breakdown Exit Reason ─────────────────────────────────────
        exit_breakdown: Dict[str, int] = (
            df_t["exit_reason"].value_counts().to_dict()
        )

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "initial_capital_usd": round(self.initial_capital, 4),
                "final_capital_usd":   round(final_capital, 4),
                "total_net_pnl_usd":   round(total_net_pnl, 4),
                "total_return_pct":    round(total_return_pct, 2),
                "total_fees_paid_usd": round(total_fees, 4),
            },
            "trade_stats": {
                "total_trades":   total_trades,
                "winning_trades": n_win,
                "losing_trades":  n_loss,
                "win_rate_pct":   round(win_rate, 2),
                "avg_win_usd":    round(avg_win, 4),
                "avg_loss_usd":   round(avg_loss, 4),
                "profit_factor":  (
                    round(profit_factor, 4) if math.isfinite(profit_factor) else None
                ),
            },
            "risk_metrics": {
                "max_drawdown_pct":        round(max_drawdown, 2),
                "sharpe_ratio_per_candle": round(sharpe_ratio, 4),
            },
            "directional_breakdown": directional_breakdown,
            "strategy_breakdown":    strategy_breakdown,
            "exit_breakdown":        exit_breakdown,
        }

    # ==================================================================
    # Cetak Laporan Teks Rapi ke Terminal
    # ==================================================================
    def print_report(
        self,
        metrics:   Dict[str, Any],
        symbol:    str,
        timeframe: str,
        mode:      str,
    ) -> None:
        """
        Mencetak laporan performa ke terminal dalam format teks terstruktur
        diikuti output JSON lengkap untuk konsumsi AI Agent.

        Args:
            metrics:   Dictionary metrik dari calculate_metrics().
            symbol:    Simbol pasangan trading yang digunakan.
            timeframe: Timeframe candle yang digunakan.
            mode:      Mode exchange aktif ('sandbox' atau 'live').
        """
        W    = 72
        SEP  = "=" * W
        DASH = "-" * W

        print(f"\n{SEP}")
        print(f"{'LAPORAN BACKTEST ADAPTIF — LONG / SHORT ENGINE':^{W}}")
        print(SEP)
        print(f"  Tanggal Laporan  : {metrics.get('generated_at', '-')}")
        print(f"  Symbol           : {symbol}")
        print(f"  Timeframe        : {timeframe}")
        print(f"  Mode Exchange    : {mode.upper()}")
        print(SEP)

        if "error" in metrics:
            print(f"\n  ⚠  {metrics['error']}")
            print(f"{SEP}\n")
            return

        s = metrics["summary"]
        t = metrics["trade_stats"]
        r = metrics["risk_metrics"]

        # ── Ringkasan Modal ───────────────────────────────────────────
        print(f"\n  [ RINGKASAN MODAL ]")
        print(DASH)
        print(f"  Modal Awal          : ${s['initial_capital_usd']:>13,.2f}")
        print(f"  Modal Akhir         : ${s['final_capital_usd']:>13,.2f}")
        print(f"  Total Net PnL       : ${s['total_net_pnl_usd']:>+13,.4f}")
        print(f"  Total Return        :  {s['total_return_pct']:>12.2f}%")
        print(f"  Total Biaya (Fee)   : ${s['total_fees_paid_usd']:>13,.4f}")

        # ── Statistik Trade ───────────────────────────────────────────
        print(f"\n  [ STATISTIK TRADE ]")
        print(DASH)
        print(f"  Total Trades        : {t['total_trades']:>14,}")
        print(f"  Trade Menang (WIN)  : {t['winning_trades']:>14,}")
        print(f"  Trade Kalah (LOSS)  : {t['losing_trades']:>14,}")
        print(f"  Win Rate            :  {t['win_rate_pct']:>12.2f}%")
        print(f"  Rata-rata Win       : ${t['avg_win_usd']:>+13,.4f}")
        print(f"  Rata-rata Loss      : ${t['avg_loss_usd']:>+13,.4f}")
        pf_val = t['profit_factor']
        pf_str = f"{pf_val:>14.4f}" if pf_val is not None else f"{'∞':>14}"
        print(f"  Profit Factor       : {pf_str}")

        # ── Metrik Risiko ─────────────────────────────────────────────
        print(f"\n  [ METRIK RISIKO ]")
        print(DASH)
        print(f"  Max Drawdown        :  {r['max_drawdown_pct']:>12.2f}%")
        print(f"  Sharpe Ratio *      : {r['sharpe_ratio_per_candle']:>14.4f}")
        print(f"  * per-candle | Risk-Free = 0% | non-annualized")

        # ── Long vs Short Breakdown ───────────────────────────────────
        print(f"\n  [ LONG vs SHORT BREAKDOWN ]")
        print(DASH)
        # Kolom: Arah(6) Trades(7) WIN(5) LOSS(5) WinRate(9) NetPnL(15+$=16)
        print(
            f"  {'Arah':<6}  {'Trades':>7}  {'WIN':>5}  {'LOSS':>5}  "
            f"{'Win Rate':>9}  {'Net PnL':>15}"
        )
        print(
            f"  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*9}  {'─'*15}"
        )
        for direction, data in metrics.get("directional_breakdown", {}).items():
            print(
                f"  {direction:<6}  {data['total_trades']:>7,}  "
                f"{data['wins']:>5}  {data['losses']:>5}  "
                f"{data['win_rate_pct']:>8.1f}%  "
                f"${data['total_pnl_usd']:>+14,.4f}"
            )

        # ── Performa per Sub-Strategi ─────────────────────────────────
        # Lebar kolom strategi = 24 karakter:
        #   "TREND_FOLLOWING_SHORT"  = 21 karakter  ✓
        #   "MEAN_REVERSION_SHORT"   = 20 karakter  ✓
        COL_S = 24
        print(f"\n  [ PERFORMA PER STRATEGI ]")
        print(DASH)
        print(
            f"  {'Strategi':<{COL_S}}  {'Trades':>6}  {'WIN':>5}  "
            f"{'LOSS':>5}  {'Win Rate':>9}  {'Net PnL':>15}"
        )
        print(
            f"  {'─'*COL_S}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*9}  {'─'*15}"
        )
        for strat, data in metrics["strategy_breakdown"].items():
            label = strat[:COL_S]          # truncate jika melebihi batas
            print(
                f"  {label:<{COL_S}}  {data['total_trades']:>6,}  "
                f"{data['wins']:>5}  {data['losses']:>5}  "
                f"{data['win_rate_pct']:>8.1f}%  "
                f"${data['total_pnl_usd']:>+14,.4f}"
            )

        # ── Breakdown Exit ────────────────────────────────────────────
        print(f"\n  [ BREAKDOWN EXIT ]")
        print(DASH)
        for reason, count in metrics["exit_breakdown"].items():
            print(f"  {reason:<22} : {count:>5,} trade(s)")

        # ── JSON Output ───────────────────────────────────────────────
        print(f"\n{SEP}")
        print(f"{'JSON OUTPUT  (AI Agent / AionUi)':^{W}}")
        print(SEP)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"{SEP}\n")

    # ==================================================================
    # Cetak Daftar Trade Detail
    # ==================================================================
    def print_trade_list(
        self,
        trades:   List[Dict[str, Any]],
        max_rows: int = 25,
    ) -> None:
        """
        Mencetak daftar trade secara detail ke terminal.

        Kolom DIR menampilkan arah posisi (LONG / SHRT) dari engine v3.
        Kolom Strategi diperlebar ke 24 karakter untuk label panjang.

        Args:
            trades:   Daftar catatan trade dari BacktestEngine.
            max_rows: Jumlah trade terakhir yang ditampilkan.
        """
        if not trades:
            print("  [!] Tidak ada trade untuk ditampilkan.")
            return

        display = trades[-max_rows:]
        W_LINE  = 122

        print(f"\n  [ DAFTAR TRADE TERAKHIR  (maks {max_rows} trade) ]")
        print(f"  {'─' * W_LINE}")
        print(
            f"  {'#':>3}  {'Entry Time':^20}  {'Strategi':<24}  {'DIR':<4}  "
            f"{'Entry':>10}  {'Exit':>10}  {'SL':>10}  {'TP':>10}  "
            f"{'Net PnL':>10}  {'Hasil':<5}"
        )
        print(f"  {'─' * W_LINE}")

        for i, tr in enumerate(display, start=1):
            # Kompatibel mundur: engine < v3 tidak punya 'position_type'
            pos_type  = tr.get("position_type", "LONG")
            dir_label = "SHRT" if pos_type == "SHORT" else "LONG"
            strat_col = tr["strategy"][:24]
            print(
                f"  {i:>3}  {str(tr['entry_time'])[:20]:^20}  "
                f"{strat_col:<24}  {dir_label:<4}  "
                f"${tr['entry_price']:>9,.2f}  "
                f"${tr['exit_price']:>9,.2f}  "
                f"${tr['stop_loss']:>9,.2f}  "
                f"${tr['take_profit']:>9,.2f}  "
                f"${tr['net_pnl']:>+9,.4f}  "
                f"{tr['outcome']:<5}"
            )

        print(f"  {'─' * W_LINE}\n")



class PerformanceReporter:
    """
    Menghitung metrik dan mencetak laporan hasil backtesting.

    Mendukung output ke terminal (teks rapi) dan JSON untuk integrasi
    dengan AI Agent atau sistem monitoring eksternal (AionUi).
    """

    def __init__(self, initial_capital: float) -> None:
        """
        Inisialisasi reporter.

        Args:
            initial_capital: Modal awal yang digunakan dalam simulasi.
        """
        self.initial_capital = initial_capital

    # ==================================================================
    # Kalkulasi Metrik
    # ==================================================================
    def calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        final_capital: float,
    ) -> Dict[str, Any]:
        """
        Menghitung semua metrik performa dari hasil backtest.

        Args:
            trades:        Daftar catatan trade dari BacktestEngine.run().
            equity_curve:  Kurva ekuitas mark-to-market per candle.
            final_capital: Modal akhir setelah semua posisi ditutup.

        Returns:
            Dictionary terstruktur berisi semua metrik performa.
        """
        if not trades:
            logger.warning("[Reporter] Tidak ada trade yang terjadi selama periode backtest.")
            return {
                "error": "Tidak ada trade yang dieksekusi selama periode backtest."
            }

        df_t = pd.DataFrame(trades)

        # ── Metrik Dasar ─────────────────────────────────────────────
        total_trades   = len(df_t)
        n_win          = int((df_t["outcome"] == "WIN").sum())
        n_loss         = int((df_t["outcome"] == "LOSS").sum())
        win_rate       = (n_win / total_trades * 100) if total_trades > 0 else 0.0

        # ── PnL ───────────────────────────────────────────────────────
        total_net_pnl    = float(df_t["net_pnl"].sum())
        total_return_pct = (total_net_pnl / self.initial_capital) * 100
        total_fees       = float(df_t["total_fees"].sum())

        wins_df   = df_t[df_t["outcome"] == "WIN"]["net_pnl"]
        losses_df = df_t[df_t["outcome"] == "LOSS"]["net_pnl"]

        avg_win  = float(wins_df.mean())   if len(wins_df) > 0   else 0.0
        avg_loss = float(losses_df.mean()) if len(losses_df) > 0 else 0.0

        # ── Profit Factor ─────────────────────────────────────────────
        gross_profit = float(df_t[df_t["net_pnl"] > 0]["net_pnl"].sum())
        gross_loss   = abs(float(df_t[df_t["net_pnl"] < 0]["net_pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # ── Maximum Drawdown (Peak-to-Trough) ────────────────────────
        equity_series = pd.Series(equity_curve, dtype=float)
        rolling_peak  = equity_series.expanding().max()
        drawdown_pct  = ((equity_series - rolling_peak) / rolling_peak) * 100
        max_drawdown  = float(drawdown_pct.min())   # nilai negatif

        # ── Breakdown per Strategi ────────────────────────────────────
        strategy_breakdown: Dict[str, Any] = {}
        for strat in df_t["strategy"].unique():
            sdf   = df_t[df_t["strategy"] == strat]
            s_win = int((sdf["outcome"] == "WIN").sum())
            strategy_breakdown[strat] = {
                "total_trades":  len(sdf),
                "wins":          s_win,
                "losses":        len(sdf) - s_win,
                "win_rate_pct":  round((s_win / len(sdf)) * 100, 2),
                "total_pnl_usd": round(float(sdf["net_pnl"].sum()), 4),
            }

        # ── Breakdown Exit Reason ─────────────────────────────────────
        exit_breakdown: Dict[str, int] = (
            df_t["exit_reason"].value_counts().to_dict()
        )

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "initial_capital_usd": round(self.initial_capital, 4),
                "final_capital_usd":   round(final_capital, 4),
                "total_net_pnl_usd":   round(total_net_pnl, 4),
                "total_return_pct":    round(total_return_pct, 2),
                "total_fees_paid_usd": round(total_fees, 4),
            },
            "trade_stats": {
                "total_trades":    total_trades,
                "winning_trades":  n_win,
                "losing_trades":   n_loss,
                "win_rate_pct":    round(win_rate, 2),
                "avg_win_usd":     round(avg_win, 4),
                "avg_loss_usd":    round(avg_loss, 4),
                "profit_factor":   round(profit_factor, 4),
            },
            "risk_metrics": {
                "max_drawdown_pct": round(max_drawdown, 2),
            },
            "strategy_breakdown": strategy_breakdown,
            "exit_breakdown":     exit_breakdown,
        }

    # ==================================================================
    # Cetak Laporan Teks Rapi ke Terminal
    # ==================================================================
    def print_report(
        self,
        metrics: Dict[str, Any],
        symbol: str,
        timeframe: str,
        mode: str,
    ) -> None:
        """
        Mencetak laporan performa ke terminal dalam format teks terstruktur
        diikuti output JSON lengkap untuk konsumsi AI Agent / AionUi.

        Args:
            metrics:   Dictionary metrik dari calculate_metrics().
            symbol:    Simbol pasangan trading yang digunakan.
            timeframe: Timeframe candle yang digunakan.
            mode:      Mode exchange aktif ('sandbox' atau 'live').
        """
        W  = 68
        SEP  = "=" * W
        DASH = "-" * W

        print(f"\n{SEP}")
        print(f"{'LAPORAN BACKTEST ADAPTIF — DYNAMIC STRATEGY SWITCHING':^{W}}")
        print(SEP)
        print(f"  Tanggal Laporan  : {metrics.get('generated_at', '-')}")
        print(f"  Symbol           : {symbol}")
        print(f"  Timeframe        : {timeframe}")
        print(f"  Mode Exchange    : {mode.upper()}")
        print(SEP)

        if "error" in metrics:
            print(f"\n  ⚠  {metrics['error']}")
            print(f"{SEP}\n")
            return

        s = metrics["summary"]
        t = metrics["trade_stats"]
        r = metrics["risk_metrics"]

        # ── Ringkasan Modal ───────────────────────────────────────────
        print(f"\n  [ RINGKASAN MODAL ]")
        print(DASH)
        print(f"  Modal Awal          : ${s['initial_capital_usd']:>13,.2f}")
        print(f"  Modal Akhir         : ${s['final_capital_usd']:>13,.2f}")
        print(f"  Total Net PnL       : ${s['total_net_pnl_usd']:>+13,.4f}")
        print(f"  Total Return        :  {s['total_return_pct']:>12.2f}%")
        print(f"  Total Biaya (Fee)   : ${s['total_fees_paid_usd']:>13,.4f}")

        # ── Statistik Trade ───────────────────────────────────────────
        print(f"\n  [ STATISTIK TRADE ]")
        print(DASH)
        print(f"  Total Trades        : {t['total_trades']:>14}")
        print(f"  Trade Menang (WIN)  : {t['winning_trades']:>14}")
        print(f"  Trade Kalah (LOSS)  : {t['losing_trades']:>14}")
        print(f"  Win Rate            :  {t['win_rate_pct']:>12.2f}%")
        print(f"  Rata-rata Win       : ${t['avg_win_usd']:>+13,.4f}")
        print(f"  Rata-rata Loss      : ${t['avg_loss_usd']:>+13,.4f}")
        print(f"  Profit Factor       : {t['profit_factor']:>14.4f}")

        # ── Metrik Risiko ─────────────────────────────────────────────
        print(f"\n  [ METRIK RISIKO ]")
        print(DASH)
        print(f"  Max Drawdown        :  {r['max_drawdown_pct']:>12.2f}%")

        # ── Performa per Strategi ─────────────────────────────────────
        print(f"\n  [ PERFORMA PER STRATEGI ]")
        print(DASH)
        hdr = f"  {'Strategi':<22} {'Trade':>6} {'WIN':>6} {'LOSS':>6} {'WR%':>8} {'Net PnL':>14}"
        print(hdr)
        print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*14}")
        for strat, data in metrics["strategy_breakdown"].items():
            print(
                f"  {strat:<22} {data['total_trades']:>6} "
                f"{data['wins']:>6} {data['losses']:>6} "
                f"{data['win_rate_pct']:>7.1f}% "
                f"${data['total_pnl_usd']:>+13,.4f}"
            )

        # ── Breakdown Exit ────────────────────────────────────────────
        print(f"\n  [ BREAKDOWN EXIT ]")
        print(DASH)
        for reason, count in metrics["exit_breakdown"].items():
            print(f"  {reason:<22} : {count:>4} trade(s)")

        # ── JSON Output ───────────────────────────────────────────────
        print(f"\n{SEP}")
        print(f"{'JSON OUTPUT  (AI Agent / AionUi)':^{W}}")
        print(SEP)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"{SEP}\n")

    # ==================================================================
    # Cetak Daftar Trade Detail
    # ==================================================================
    def print_trade_list(
        self,
        trades: List[Dict[str, Any]],
        max_rows: int = 25,
    ) -> None:
        """
        Mencetak daftar trade secara detail ke terminal.

        Args:
            trades:   Daftar catatan trade dari BacktestEngine.
            max_rows: Jumlah trade terakhir yang ditampilkan.
        """
        if not trades:
            print("  [!] Tidak ada trade untuk ditampilkan.")
            return

        display = trades[-max_rows:]
        W_LINE  = 106

        print(f"\n  [ DAFTAR TRADE TERAKHIR  (maks {max_rows} trade) ]")
        print(f"  {'─' * W_LINE}")
        print(
            f"  {'#':>3}  {'Entry Time':^20}  {'Strategi':<16}  "
            f"{'Entry':>10}  {'Exit':>10}  {'SL':>10}  {'TP':>10}  "
            f"{'Net PnL':>10}  {'Hasil':<5}"
        )
        print(f"  {'─' * W_LINE}")

        for i, tr in enumerate(display, start=1):
            print(
                f"  {i:>3}  {str(tr['entry_time'])[:20]:^20}  "
                f"{tr['strategy']:<16}  "
                f"${tr['entry_price']:>9,.2f}  "
                f"${tr['exit_price']:>9,.2f}  "
                f"${tr['stop_loss']:>9,.2f}  "
                f"${tr['take_profit']:>9,.2f}  "
                f"${tr['net_pnl']:>+9,.4f}  "
                f"{tr['outcome']:<5}"
            )

        print(f"  {'─' * W_LINE}\n")
