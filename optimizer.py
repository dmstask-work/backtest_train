"""
Grid Search Optimizer — Pencari Parameter Terbaik secara Otomatis.

Mengeksekusi backtest untuk setiap kombinasi parameter yang ditentukan
dan menampilkan 5 kombinasi terbaik berdasarkan Net PnL.

Cara menjalankan:
  python optimizer.py

Opsi kustomisasi ada di bagian OPTIMIZER CONFIG di bawah ini.
Tidak perlu mengubah kode lainnya.

Alur Kerja:
  1. Ambil data OHLCV SATU kali (hemat kuota API)
  2. Buat semua kombinasi parameter (grid) via itertools.product
  3. Untuk setiap kombinasi:
       a. Hitung ulang indikator (vectorized, sangat cepat)
       b. Klasifikasi regime (ADX)
       c. Generate sinyal Trend-Following ONLY
       d. Jalankan BacktestEngine
       e. Catat metrik
  4. Urutkan hasil, cetak Top 5
"""

import itertools
import logging
import sys
import time
from copy import deepcopy
from typing import Any

import pandas as pd

# Impor modul internal
from backtest_engine import BacktestEngine
from config import BACKTEST_CONFIG, EXCHANGE_MODE, INDICATOR_CONFIG, RISK_CONFIG
from data_fetcher import DataFetcher
from indicators import IndicatorCalculator
from regime_filter import RegimeFilter
from reporter import PerformanceReporter
from risk_manager import RiskManager
from signal_generator import SignalGenerator

# ============================================================
# OPTIMIZER CONFIG — Edit bagian ini untuk kustomisasi
# ============================================================

# ── Data Source ───────────────────────────────────────────────────────
OPT_SYMBOL:    str = "SOL/USDT"
OPT_TIMEFRAME: str = "4h"
OPT_LIMIT:     int = 8000        # Total candle historis (diambil 1x)
OPT_MODE:      str = EXCHANGE_MODE

# ── Parameter Backtest Tetap ──────────────────────────────────────────
OPT_INITIAL_CAPITAL:  float = BACKTEST_CONFIG["initial_capital"]
OPT_TRADE_ALLOCATION: float = 0.40   # 40% per trade
OPT_FEE_RATE:         float = BACKTEST_CONFIG["fee_rate"]

# ── Rentang Parameter yang Dioptimasi (Grid) ──────────────────────────
PARAM_GRID: dict = {
    "ema_fast":      [10, 15, 20, 25],
    "ema_slow":      [40, 50, 60, 100],
    "adx_threshold": [20, 25, 30],
}

# ── Output ────────────────────────────────────────────────────────────
TOP_N:        int  = 5       # Tampilkan N kombinasi terbaik
SORT_BY:      str  = "net_pnl"  # Metrik pengurutan: 'net_pnl' | 'win_rate'
SHOW_ALL:     bool = False   # True = tampilkan semua hasil, bukan hanya Top N

# ============================================================
# Setup Logging
# ============================================================
logging.basicConfig(
    level=logging.WARNING,   # Hanya tampilkan WARNING+ agar output bersih
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Logger khusus optimizer pada level INFO agar progress tetap terlihat
opt_logger = logging.getLogger("optimizer")
opt_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
)
opt_logger.addHandler(_handler)
opt_logger.propagate = False


# ============================================================
# Fungsi Helper
# ============================================================
def _build_indicator_config(overrides: dict) -> dict:
    """
    Membuat salinan INDICATOR_CONFIG dengan nilai yang di-override.

    Menggunakan deepcopy agar base config tidak termutasi selama looping.

    Args:
        overrides: Dictionary parameter yang akan ditimpa.

    Returns:
        Dictionary config indikator baru.
    """
    cfg = deepcopy(INDICATOR_CONFIG)
    cfg.update(overrides)
    return cfg


def _run_single_backtest(
    df_raw: pd.DataFrame,
    indicator_cfg: dict,
    adx_threshold: float,
) -> dict[str, Any]:
    """
    Menjalankan satu siklus backtest penuh untuk satu kombinasi parameter.

    Args:
        df_raw:          DataFrame OHLCV mentah (sebelum kalkulasi indikator).
        indicator_cfg:   Config indikator yang sudah di-override.
        adx_threshold:   Nilai ADX threshold untuk regime filter.

    Returns:
        Dictionary metrik performa: net_pnl, win_rate, max_drawdown,
        total_trades, final_capital. Atau dict berisi 'error' jika gagal.
    """
    try:
        # ── Hitung Indikator (vectorized) ─────────────────────────────
        calculator = IndicatorCalculator(config=indicator_cfg)
        df = calculator.calculate_all(df_raw)

        # ── Jika data terlalu sedikit setelah warm-up, skip ───────────
        if len(df) < 50:
            return {"error": "data_terlalu_sedikit"}

        # ── Regime Filter ─────────────────────────────────────────────
        regime_filter = RegimeFilter(adx_threshold=adx_threshold)
        df = regime_filter.classify(df)

        # ── Signal Generator: ONLY Trend-Following ────────────────────
        signal_gen = SignalGenerator(
            config=indicator_cfg,
            enable_trend_following=True,
            enable_mean_reversion=False,   # DIMATIKAN sesuai spesifikasi
        )
        df = signal_gen.generate(df)

        # ── Backtest Engine ───────────────────────────────────────────
        risk_manager = RiskManager(
            sl_multiplier=RISK_CONFIG["sl_multiplier"],
            tp_multiplier=RISK_CONFIG["tp_multiplier"],
        )
        engine = BacktestEngine(
            initial_capital  = OPT_INITIAL_CAPITAL,
            trade_allocation = OPT_TRADE_ALLOCATION,
            fee_rate         = OPT_FEE_RATE,
            risk_manager     = risk_manager,
        )
        results = engine.run(df)

        # ── Kalkulasi Metrik ──────────────────────────────────────────
        reporter = PerformanceReporter(initial_capital=OPT_INITIAL_CAPITAL)
        metrics  = reporter.calculate_metrics(
            trades       = results["trades"],
            equity_curve = results["equity_curve"],
            final_capital= results["final_capital"],
        )

        if "error" in metrics:
            return {"error": metrics["error"]}

        return {
            "net_pnl":      metrics["summary"]["total_net_pnl_usd"],
            "win_rate":     metrics["trade_stats"]["win_rate_pct"],
            "max_drawdown": metrics["risk_metrics"]["max_drawdown_pct"],
            "total_trades": metrics["trade_stats"]["total_trades"],
            "profit_factor": metrics["trade_stats"]["profit_factor"],
            "final_capital": metrics["summary"]["final_capital_usd"],
        }

    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ============================================================
# Validasi Grid (filter kombinasi tidak logis)
# ============================================================
def _is_valid_combo(params: dict) -> bool:
    """
    Menyaring kombinasi parameter yang tidak masuk akal secara logis.

    Aturan:
      - ema_fast harus lebih kecil dari ema_slow (minimal selisih 10)
        agar ada gap yang berarti antara garis cepat dan lambat.

    Args:
        params: Dictionary satu kombinasi parameter.

    Returns:
        True jika valid, False jika harus di-skip.
    """
    return (params["ema_fast"] + 10) <= params["ema_slow"]


# ============================================================
# Fungsi Utama Optimizer
# ============================================================
def run_optimizer() -> pd.DataFrame:
    """
    Menjalankan Grid Search secara penuh dan mengembalikan DataFrame hasil.

    Returns:
        DataFrame semua kombinasi valid yang berhasil diuji,
        diurutkan berdasarkan SORT_BY (default: net_pnl) secara descending.
    """
    W = 70

    print(f"\n{'=' * W}")
    print(f"{'GRID SEARCH OPTIMIZER — ADAPTIVE BACKTEST SYSTEM':^{W}}")
    print(f"{'=' * W}")
    print(f"  Symbol      : {OPT_SYMBOL}")
    print(f"  Timeframe   : {OPT_TIMEFRAME}")
    print(f"  Candle Limit: {OPT_LIMIT:,}")
    print(f"  Alokasi     : {OPT_TRADE_ALLOCATION * 100:.0f}% per trade")
    print(f"  Modal Awal  : ${OPT_INITIAL_CAPITAL:,.2f}")
    print(f"  Strategi    : TREND-FOLLOWING ONLY")
    print(f"  Grid        : ema_fast {PARAM_GRID['ema_fast']}")
    print(f"              : ema_slow {PARAM_GRID['ema_slow']}")
    print(f"              : adx_threshold {PARAM_GRID['adx_threshold']}")
    print(f"{'=' * W}\n")

    # ── FASE 1: Ambil Data SATU KALI ─────────────────────────────────
    opt_logger.info("[FASE 1] Mengambil %d candle %s %s ...",
                    OPT_LIMIT, OPT_SYMBOL, OPT_TIMEFRAME)
    t_fetch_start = time.perf_counter()

    fetcher = DataFetcher(mode=OPT_MODE)
    df_raw  = fetcher.fetch_ohlcv(
        symbol    = OPT_SYMBOL,
        timeframe = OPT_TIMEFRAME,
        limit     = OPT_LIMIT,
    )

    t_fetch_elapsed = time.perf_counter() - t_fetch_start
    opt_logger.info(
        "[FASE 1] Data siap: %d baris dalam %.1f detik.",
        len(df_raw), t_fetch_elapsed,
    )

    # ── FASE 2: Bangun Grid Semua Kombinasi ───────────────────────────
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combos = [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]

    # Filter kombinasi tidak logis (ema_fast >= ema_slow - 10)
    valid_combos = [c for c in all_combos if _is_valid_combo(c)]
    skipped      = len(all_combos) - len(valid_combos)

    total = len(valid_combos)
    opt_logger.info(
        "[FASE 2] Grid: %d total kombinasi | %d valid | %d di-skip (ema_fast ≥ ema_slow-10)",
        len(all_combos), total, skipped,
    )

    # ── FASE 3: Loop Backtest ─────────────────────────────────────────
    opt_logger.info("[FASE 3] Memulai iterasi backtest ...\n")

    results_log: list[dict[str, Any]] = []
    t_loop_start = time.perf_counter()

    for i, params in enumerate(valid_combos, start=1):
        ema_f  = params["ema_fast"]
        ema_s  = params["ema_slow"]
        adx_th = params["adx_threshold"]

        # Progress indicator
        pct = (i / total) * 100
        print(
            f"  [{i:>3}/{total}] ({pct:>5.1f}%) "
            f"ema_fast={ema_f:>3} | ema_slow={ema_s:>3} | "
            f"adx_threshold={adx_th:>2} ...",
            end="  ",
            flush=True,
        )

        # Override config untuk kombinasi ini
        ind_cfg = _build_indicator_config({
            "ema_fast":      ema_f,
            "ema_slow":      ema_s,
            "adx_threshold": adx_th,
        })

        t0      = time.perf_counter()
        outcome = _run_single_backtest(df_raw, ind_cfg, adx_threshold=adx_th)
        elapsed = time.perf_counter() - t0

        if "error" in outcome:
            print(f"SKIP ({outcome['error']})")
            continue

        row = {
            "ema_fast":      ema_f,
            "ema_slow":      ema_s,
            "adx_threshold": adx_th,
            **outcome,
            "_time_s":       round(elapsed, 3),
        }
        results_log.append(row)

        # Tampilkan ringkasan per baris
        print(
            f"PnL: ${outcome['net_pnl']:>+8.2f} | "
            f"WR: {outcome['win_rate']:>5.1f}% | "
            f"DD: {outcome['max_drawdown']:>6.2f}% | "
            f"Trades: {outcome['total_trades']:>3} | "
            f"{elapsed:.2f}s"
        )

    t_loop_elapsed = time.perf_counter() - t_loop_start

    print(f"\n  Iterasi selesai dalam {t_loop_elapsed:.1f} detik total.\n")

    if not results_log:
        opt_logger.error("Tidak ada kombinasi yang menghasilkan trade. "
                         "Coba perluas rentang parameter atau kurangi filter.")
        sys.exit(1)

    # ── FASE 4: Ranking & Output ──────────────────────────────────────
    df_results = pd.DataFrame(results_log)
    df_results.sort_values(by=SORT_BY, ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)

    _print_top_results(df_results, top_n=TOP_N if not SHOW_ALL else len(df_results))

    return df_results


# ============================================================
# Fungsi Cetak Hasil
# ============================================================
def _print_top_results(df: pd.DataFrame, top_n: int) -> None:
    """
    Mencetak tabel Top N kombinasi parameter terbaik ke terminal.

    Args:
        df:    DataFrame hasil yang sudah diurutkan.
        top_n: Jumlah baris teratas yang ditampilkan.
    """
    W   = 88
    SEP = "=" * W
    label = f"TOP {top_n}" if top_n < len(df) else "SEMUA HASIL"

    print(f"\n{SEP}")
    print(f"  {label} KOMBINASI PARAMETER TERBAIK  "
          f"(diurutkan berdasarkan: {SORT_BY.upper()})".center(W))
    print(SEP)

    header = (
        f"  {'#':>3}  "
        f"{'EMA_F':>6}  {'EMA_S':>6}  {'ADX_TH':>6}  "
        f"{'Net PnL':>10}  {'WR%':>6}  {'Max DD%':>8}  "
        f"{'Trades':>7}  {'PF':>6}  {'Final $':>10}"
    )
    print(header)
    print(f"  {'─' * (W - 4)}")

    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), start=1):
        pnl_sign = "+" if row["net_pnl"] >= 0 else ""
        print(
            f"  {rank:>3}  "
            f"{int(row['ema_fast']):>6}  {int(row['ema_slow']):>6}  "
            f"{int(row['adx_threshold']):>6}  "
            f"${pnl_sign}{row['net_pnl']:>9.2f}  "
            f"{row['win_rate']:>5.1f}%  "
            f"{row['max_drawdown']:>7.2f}%  "
            f"{int(row['total_trades']):>7}  "
            f"{row['profit_factor']:>6.2f}  "
            f"${row['final_capital']:>9.2f}"
        )

    print(SEP)

    # Cetak #1 Terbaik secara detail
    best = df.iloc[0]
    print(f"\n  ★  REKOMENDASI TERBAIK:")
    print(f"     ema_fast={int(best['ema_fast'])} | "
          f"ema_slow={int(best['ema_slow'])} | "
          f"adx_threshold={int(best['adx_threshold'])}")
    print(f"     Net PnL  : ${best['net_pnl']:+.4f}")
    print(f"     Win Rate : {best['win_rate']:.2f}%")
    print(f"     Max DD   : {best['max_drawdown']:.2f}%")
    print(f"     Trades   : {int(best['total_trades'])}")
    print(f"     PF       : {best['profit_factor']:.4f}")
    print(f"\n     → Untuk menerapkan: edit settings.json atau jalankan:")
    print(
        f"     python main.py --symbol {OPT_SYMBOL} --tf {OPT_TIMEFRAME} "
        f"--alloc {OPT_TRADE_ALLOCATION} --no-meanrev"
    )
    print(f"       (dan set ema_fast={int(best['ema_fast'])}, "
          f"ema_slow={int(best['ema_slow'])}, "
          f"adx_threshold={int(best['adx_threshold'])} di settings.json)")
    print(f"{SEP}\n")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    run_optimizer()
