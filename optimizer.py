"""
Grid Search Optimizer V3 — Pencari Parameter Terbaik untuk Long/Short Engine.

Mengeksekusi backtest untuk setiap kombinasi parameter yang ditentukan
dan menampilkan Top N kombinasi terbaik berdasarkan metrik pilihan
(default: Sharpe Ratio per-candle — prioritas risk-adjusted return).

Cara menjalankan:
  python optimizer.py                                    # mode interaktif biasa
  python optimizer.py --symbol BTC/USDT --auto-apply    # override + tulis settings
  python optimizer.py --symbol ETH/USDT --timeframe 1h --limit 5000

Argumen CLI (semua opsional):
  --symbol      Override trading pair  (contoh: BTC/USDT)
  --timeframe   Override timeframe     (contoh: 1h, 4h, 1d)
  --limit       Override candle limit  (contoh: 5000)
  --auto-apply  Jika di-set, parameter terbaik langsung ditulis ke settings.json

Opsi kustomisasi statis ada di bagian OPTIMIZER CONFIG di bawah ini.

Pembaruan V3 (dibanding V2):
  • Kedua strategi AKTIF: Trend-Following + Mean-Reversion (regime switching)
  • Slippage dikonfigurasikan eksplisit (sesuai BacktestEngine v3)
  • Grid diperluas: bb_std dan rsi_oversold masuk ke dalam pencarian
  • Metrik Sharpe Ratio per-candle ditangkap dan ditampilkan
  • Directional breakdown (Long WR% vs Short WR%) tersedia di rekomendasi
  • SORT_BY default: 'sharpe_ratio' (risk-adjusted, bukan nominal PnL)

Alur Kerja:
  1. Ambil data OHLCV SATU kali (hemat kuota API)
  2. Buat semua kombinasi parameter (grid) via itertools.product
  3. Untuk setiap kombinasi yang valid:
       a. Hitung ulang indikator (vectorized, sangat cepat)
       b. Klasifikasi regime (ADX threshold)
       c. Generate sinyal LONG + SHORT via kedua strategi
       d. Jalankan BacktestEngine v3 (slippage + pending order)
       e. Catat metrik: PnL, Sharpe, WR, DD, PF, Long/Short breakdown
  4. Urutkan hasil, cetak Top N
"""

import argparse
import itertools
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

# ── Impor modul internal ─────────────────────────────────────────────────────
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
OPT_LIMIT:     int = 8000          # Total candle historis (diambil 1×)
OPT_MODE:      str = EXCHANGE_MODE

# ── Parameter Backtest Tetap (non-grid) ───────────────────────────────
OPT_INITIAL_CAPITAL:  float = BACKTEST_CONFIG["initial_capital"]
OPT_TRADE_ALLOCATION: float = 0.40     # 40% per trade
OPT_FEE_RATE:         float = BACKTEST_CONFIG["fee_rate"]
OPT_SLIPPAGE_RATE:    float = 0.0005   # Konsisten dengan BacktestEngine v3

# ── Rentang Parameter yang Dioptimasi (Grid) ──────────────────────────
# Total kombinasi: 4 × 4 × 3 × 2 × 2 = 192 (sebelum filter EMA)
PARAM_GRID: dict = {
    "ema_fast":      [10, 15, 20, 25],      # Periode EMA cepat
    "ema_slow":      [40, 50, 60, 100],     # Periode EMA lambat
    "adx_threshold": [20, 25, 30],          # Batas ADX untuk regime TRENDING
    "bb_std":        [2.0, 2.5],            # Std dev Bollinger Bands
    "rsi_oversold":  [25, 30],              # Batas RSI oversold (MR LONG)
}

# ── Output ────────────────────────────────────────────────────────────
TOP_N:    int  = 5       # Tampilkan N kombinasi terbaik
SORT_BY:  str  = "sharpe_ratio"  # "sharpe_ratio" | "net_pnl" | "win_rate"
SHOW_ALL: bool = False   # True = tampilkan semua hasil, bukan hanya Top N

# ============================================================
# Setup Logging — Hanya optimizer yang INFO, semua modul diam
# ============================================================
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
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

    Menggunakan deepcopy agar base config tidak termutasi selama loop.

    Args:
        overrides: Dictionary parameter yang akan menimpa nilai default.

    Returns:
        Dictionary config indikator baru untuk satu kombinasi grid.
    """
    cfg = deepcopy(INDICATOR_CONFIG)
    cfg.update(overrides)
    return cfg


def _run_single_backtest(
    df_raw:        pd.DataFrame,
    indicator_cfg: dict,
    adx_threshold: float,
) -> dict[str, Any]:
    """
    Menjalankan satu siklus backtest penuh untuk satu kombinasi parameter.

    Menggunakan kedua strategi (Trend-Following + Mean-Reversion) untuk
    mensimulasikan sistem regime-switching yang lengkap.

    Args:
        df_raw:        DataFrame OHLCV mentah (sebelum kalkulasi indikator).
        indicator_cfg: Config indikator yang sudah di-override untuk kombinasi ini.
        adx_threshold: Nilai ADX threshold untuk regime filter.

    Returns:
        Dictionary metrik lengkap termasuk Sharpe dan directional breakdown,
        atau dict berisi 'error' jika backtest gagal / tidak menghasilkan trade.
    """
    try:
        # ── Hitung Indikator (vectorized) ─────────────────────────────
        calculator = IndicatorCalculator(config=indicator_cfg)
        df = calculator.calculate_all(df_raw)

        if len(df) < 50:
            return {"error": "data_terlalu_sedikit"}

        # ── Regime Filter ─────────────────────────────────────────────
        regime_filter = RegimeFilter(adx_threshold=adx_threshold)
        df = regime_filter.classify(df)

        # ── Signal Generator: KEDUA Strategi Aktif ───────────────────
        # Trend-Following (LONG & SHORT) + Mean-Reversion (LONG & SHORT)
        signal_gen = SignalGenerator(
            config                = indicator_cfg,
            enable_trend_following= True,
            enable_mean_reversion = True,
        )
        df = signal_gen.generate(df)

        # ── Backtest Engine v3 ────────────────────────────────────────
        risk_manager = RiskManager(
            sl_multiplier = RISK_CONFIG["sl_multiplier"],
            tp_multiplier = RISK_CONFIG["tp_multiplier"],
        )
        engine = BacktestEngine(
            initial_capital  = OPT_INITIAL_CAPITAL,
            trade_allocation = OPT_TRADE_ALLOCATION,
            fee_rate         = OPT_FEE_RATE,
            slippage_rate    = OPT_SLIPPAGE_RATE,   # eksplisit v3
            risk_manager     = risk_manager,
        )
        results = engine.run(df)

        # ── Kalkulasi Metrik ──────────────────────────────────────────
        reporter = PerformanceReporter(initial_capital=OPT_INITIAL_CAPITAL)
        metrics  = reporter.calculate_metrics(
            trades        = results["trades"],
            equity_curve  = results["equity_curve"],
            final_capital = results["final_capital"],
        )

        if "error" in metrics:
            return {"error": metrics["error"]}

        # ── Ekstrak Directional Breakdown ─────────────────────────────
        dir_bd  = metrics.get("directional_breakdown", {})
        long_d  = dir_bd.get("LONG",  {})
        short_d = dir_bd.get("SHORT", {})

        pf_raw = metrics["trade_stats"]["profit_factor"]

        return {
            "net_pnl":        metrics["summary"]["total_net_pnl_usd"],
            "win_rate":       metrics["trade_stats"]["win_rate_pct"],
            "sharpe_ratio":   metrics["risk_metrics"]["sharpe_ratio_per_candle"],
            "max_drawdown":   metrics["risk_metrics"]["max_drawdown_pct"],
            "total_trades":   metrics["trade_stats"]["total_trades"],
            "profit_factor":  pf_raw if pf_raw is not None else 0.0,
            "final_capital":  metrics["summary"]["final_capital_usd"],
            # ── Directional ──────────────────────────────────────────
            "long_trades":    long_d.get("total_trades",  0),
            "long_win_rate":  long_d.get("win_rate_pct",  0.0),
            "short_trades":   short_d.get("total_trades", 0),
            "short_win_rate": short_d.get("win_rate_pct", 0.0),
        }

    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ============================================================
# Validasi Grid
# ============================================================
def _is_valid_combo(params: dict) -> bool:
    """
    Menyaring kombinasi parameter yang tidak masuk akal secara logis.

    Aturan:
      - ema_fast + 10 ≤ ema_slow: gap minimal 10 periode antara EMA
        cepat dan lambat agar sinyal tren memiliki makna yang jelas.

    Args:
        params: Dictionary satu kombinasi parameter.

    Returns:
        True jika valid, False jika harus di-skip.
    """
    return (params["ema_fast"] + 10) <= params["ema_slow"]


# ============================================================
# Auto-Apply — Tulis Parameter Terbaik ke settings.json
# ============================================================

# Path settings.json relatif terhadap file ini (bukan cwd)
_SETTINGS_PATH = Path(__file__).parent / "settings.json"

# Key indikator yang dikelola oleh optimizer (whitelist)
_GRID_PARAM_KEYS: tuple[str, ...] = (
    "ema_fast",
    "ema_slow",
    "adx_threshold",
    "bb_std",
    "rsi_oversold",
)


def _apply_best_params(best: "pd.Series") -> None:
    """
    Tulis parameter terbaik dari hasil optimizer ke dalam settings.json.

    Hanya key yang terdaftar di _GRID_PARAM_KEYS yang diperbarui di dalam
    objek ``"indicators"``. Semua bagian lain (exchange, backtest, risk,
    _comment*, nested objects) dijaga utuh persis seperti semula.

    File disimpan kembali dengan indent=2 (format standar).

    Args:
        best: Row pertama dari df_results (kombinasi parameter terbaik).
    """
    # ── Muat JSON yang ada, atau mulai dari dict kosong ───────────────
    if _SETTINGS_PATH.exists():
        try:
            data: dict = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            opt_logger.error(
                "Auto-apply GAGAL: tidak dapat mem-parse settings.json — %s", exc
            )
            return
    else:
        data = {}

    # ── Pastikan section "indicators" ada ────────────────────────────
    if "indicators" not in data:
        data["indicators"] = {}

    # ── Update hanya key yang dikelola optimizer ──────────────────────
    winning: dict = {
        "ema_fast":      int(best["ema_fast"]),
        "ema_slow":      int(best["ema_slow"]),
        "adx_threshold": int(best["adx_threshold"]),
        "bb_std":        float(best["bb_std"]),
        "rsi_oversold":  int(best["rsi_oversold"]),
    }
    for key, value in winning.items():
        data["indicators"][key] = value

    # ── Simpan kembali (preserves semua bagian lain + _comment keys) ──
    _SETTINGS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ── Konfirmasi terminal ───────────────────────────────────────────
    W   = 96
    SEP = "=" * W
    print(f"\n{SEP}")
    print(f"  AUTO-APPLY BERHASIL".center(W))
    print(SEP)
    print(f"  settings.json diperbarui → {_SETTINGS_PATH}")
    print(f"  Parameter yang ditulis ke \"indicators\":")
    for key, value in winning.items():
        print(f"    {key:<20} = {value}")
    print(f"{SEP}\n")


# ============================================================
# Fungsi Utama Optimizer
# ============================================================
def run_optimizer(auto_apply: bool = False) -> pd.DataFrame:
    """
    Menjalankan Grid Search secara penuh dan mengembalikan DataFrame hasil.

    Returns:
        DataFrame semua kombinasi valid yang berhasil diuji,
        diurutkan berdasarkan SORT_BY secara descending.
    """
    W = 76

    print(f"\n{'=' * W}")
    print(f"{'GRID SEARCH OPTIMIZER V3 — LONG/SHORT REGIME-SWITCHING ENGINE':^{W}}")
    print(f"{'=' * W}")
    print(f"  Symbol        : {OPT_SYMBOL}")
    print(f"  Timeframe     : {OPT_TIMEFRAME}")
    print(f"  Candle Limit  : {OPT_LIMIT:,}")
    print(f"  Alokasi       : {OPT_TRADE_ALLOCATION * 100:.0f}% per trade")
    print(f"  Modal Awal    : ${OPT_INITIAL_CAPITAL:,.2f}")
    print(f"  Slippage      : {OPT_SLIPPAGE_RATE * 100:.2f}% per sisi")
    print(f"  Strategi      : TREND-FOLLOWING + MEAN-REVERSION  (regime switching)")
    print(f"  Urut berdasar : {SORT_BY.upper()}")
    print(f"  Grid:")
    print(f"    ema_fast      : {PARAM_GRID['ema_fast']}")
    print(f"    ema_slow      : {PARAM_GRID['ema_slow']}")
    print(f"    adx_threshold : {PARAM_GRID['adx_threshold']}")
    print(f"    bb_std        : {PARAM_GRID['bb_std']}")
    print(f"    rsi_oversold  : {PARAM_GRID['rsi_oversold']}")
    print(f"{'=' * W}\n")

    # ── FASE 1: Ambil Data SATU KALI ─────────────────────────────────
    opt_logger.info(
        "[FASE 1] Mengambil %d candle %s %s ...",
        OPT_LIMIT, OPT_SYMBOL, OPT_TIMEFRAME,
    )
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

    valid_combos = [c for c in all_combos if _is_valid_combo(c)]
    skipped      = len(all_combos) - len(valid_combos)
    total        = len(valid_combos)

    opt_logger.info(
        "[FASE 2] Grid: %d total kombinasi | %d valid | %d di-skip (gap EMA < 10)",
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
        bb_s   = params["bb_std"]
        rsi_os = params["rsi_oversold"]

        pct = (i / total) * 100
        print(
            f"  [{i:>3}/{total}] ({pct:>5.1f}%)  "
            f"ef={ema_f:>2} es={ema_s:>3} adx={adx_th:>2} "
            f"bs={bb_s:.1f} ro={rsi_os:>2}  ...",
            end="  ",
            flush=True,
        )

        ind_cfg = _build_indicator_config({
            "ema_fast":      ema_f,
            "ema_slow":      ema_s,
            "adx_threshold": adx_th,
            "bb_std":        bb_s,
            "rsi_oversold":  rsi_os,
        })

        t0      = time.perf_counter()
        outcome = _run_single_backtest(df_raw, ind_cfg, adx_threshold=adx_th)
        elapsed = time.perf_counter() - t0

        if "error" in outcome:
            print(f"SKIP  ({outcome['error']})")
            continue

        row = {
            "ema_fast":      ema_f,
            "ema_slow":      ema_s,
            "adx_threshold": adx_th,
            "bb_std":        bb_s,
            "rsi_oversold":  rsi_os,
            **outcome,
            "_time_s":       round(elapsed, 3),
        }
        results_log.append(row)

        print(
            f"PnL: ${outcome['net_pnl']:>+8.2f}  "
            f"WR: {outcome['win_rate']:>5.1f}%  "
            f"Sharpe: {outcome['sharpe_ratio']:>7.4f}  "
            f"DD: {outcome['max_drawdown']:>6.2f}%  "
            f"T: {outcome['total_trades']:>3}  "
            f"{elapsed:.2f}s"
        )

    t_loop_elapsed = time.perf_counter() - t_loop_start
    print(f"\n  Iterasi selesai dalam {t_loop_elapsed:.1f} detik total.\n")

    if not results_log:
        opt_logger.error(
            "Tidak ada kombinasi yang menghasilkan trade. "
            "Coba perluas rentang parameter atau kurangi filter."
        )
        sys.exit(1)

    # ── FASE 4: Ranking & Output ──────────────────────────────────────
    df_results = pd.DataFrame(results_log)
    df_results.sort_values(by=SORT_BY, ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)

    _print_top_results(df_results, top_n=TOP_N if not SHOW_ALL else len(df_results))

    # ── FASE 5: Auto-Apply (opsional) ────────────────────────────────
    if auto_apply:
        _apply_best_params(df_results.iloc[0])

    return df_results


# ============================================================
# Fungsi Cetak Hasil
# ============================================================
def _print_top_results(df: pd.DataFrame, top_n: int) -> None:
    """
    Mencetak tabel Top N kombinasi parameter terbaik ke terminal.

    Kolom tabel: EF, ES, ADX, BS, RO, Net PnL, WR%, Sharpe, MaxDD%, Trades, PF
    Rekomendasi terbaik ditampilkan dengan detail directional (Long/Short WR).

    Args:
        df:    DataFrame hasil yang sudah diurutkan.
        top_n: Jumlah baris teratas yang ditampilkan.
    """
    W     = 96
    SEP   = "=" * W
    label = f"TOP {top_n}" if top_n < len(df) else "SEMUA HASIL"

    print(f"\n{SEP}")
    print(
        f"  {label} KOMBINASI PARAMETER TERBAIK  "
        f"(diurutkan: {SORT_BY.upper()})".center(W)
    )
    print(SEP)

    # ── Header tabel ──────────────────────────────────────────────────
    # Kolom  : #(3) EF(4) ES(4) ADX(4) BS(5) RO(5) PnL(11) WR(6) Sharpe(8) DD(7) T(6) PF(6)
    header = (
        f"  {'#':>3}  "
        f"{'EF':>4}  {'ES':>4}  {'ADX':>4}  {'BS':>5}  {'RO':>5}  "
        f"{'Net PnL':>11}  {'WR%':>6}  {'Sharpe':>8}  "
        f"{'MaxDD%':>7}  {'Trades':>6}  {'PF':>6}"
    )
    print(header)
    print(f"  {'─' * (W - 4)}")

    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), start=1):
        pf_val  = row["profit_factor"]
        pf_str  = f"{pf_val:>6.2f}" if pf_val > 0 else f"{'0.00':>6}"
        print(
            f"  {rank:>3}  "
            f"{int(row['ema_fast']):>4}  {int(row['ema_slow']):>4}  "
            f"{int(row['adx_threshold']):>4}  "
            f"{row['bb_std']:>5.1f}  {int(row['rsi_oversold']):>5}  "
            f"${row['net_pnl']:>+10.2f}  "
            f"{row['win_rate']:>5.1f}%  "
            f"{row['sharpe_ratio']:>8.4f}  "
            f"{row['max_drawdown']:>6.2f}%  "
            f"{int(row['total_trades']):>6}  "
            f"{pf_str}"
        )

    print(SEP)

    # ── Rekomendasi Terbaik — Detail Lengkap ──────────────────────────
    best   = df.iloc[0]
    DASH_B = "─" * (W - 4)

    print(f"\n  ★  REKOMENDASI TERBAIK  (#{1} dari {len(df)} kombinasi valid)")
    print(f"  {DASH_B}")

    # Parameter
    print(f"  Parameter :")
    print(f"    ema_fast      = {int(best['ema_fast']):<5}  "
          f"ema_slow      = {int(best['ema_slow'])}")
    print(f"    adx_threshold = {int(best['adx_threshold']):<5}  "
          f"bb_std        = {best['bb_std']:.1f}")
    print(f"    rsi_oversold  = {int(best['rsi_oversold'])}")

    # Metrik agregat
    print(f"  {DASH_B}")
    print(f"  Metrik Agregat :")
    print(f"    Net PnL       : ${best['net_pnl']:>+.4f}")
    print(f"    Win Rate      :  {best['win_rate']:.2f}%")
    print(f"    Sharpe Ratio  :  {best['sharpe_ratio']:.4f}  "
          f"(per-candle | RF=0%)")
    print(f"    Max Drawdown  :  {best['max_drawdown']:.2f}%")
    print(f"    Profit Factor :  {best['profit_factor']:.4f}")
    print(f"    Total Trades  :  {int(best['total_trades'])}")
    print(f"    Final Capital : ${best['final_capital']:,.4f}")

    # Directional breakdown
    print(f"  {DASH_B}")
    print(f"  Directional Breakdown :")
    print(
        f"    LONG   — {int(best['long_trades']):>3} trade(s)  |  "
        f"Win Rate: {best['long_win_rate']:.1f}%"
    )
    print(
        f"    SHORT  — {int(best['short_trades']):>3} trade(s)  |  "
        f"Win Rate: {best['short_win_rate']:.1f}%"
    )

    # Command untuk menerapkan
    print(f"  {DASH_B}")
    print(f"  Untuk menerapkan — jalankan:")
    print(
        f"    python main.py --symbol {OPT_SYMBOL} --tf {OPT_TIMEFRAME} "
        f"--alloc {OPT_TRADE_ALLOCATION}"
    )
    print(f"  Atau terapkan otomatis ke settings.json:")
    print(
        f"    python optimizer.py --symbol {OPT_SYMBOL} "
        f"--timeframe {OPT_TIMEFRAME} --auto-apply"
    )
    print(f"{SEP}\n")


# ============================================================
# Entry Point
# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid Search Optimizer V3 — Long/Short Regime-Switching Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        metavar="PAIR",
        help="Override trading pair (default: %(default)s → gunakan OPT_SYMBOL)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        metavar="TF",
        help="Override timeframe, contoh: 1h, 4h, 1d",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Override jumlah candle historis yang diambil",
    )
    parser.add_argument(
        "--auto-apply",
        action="store_true",
        default=False,
        help="Tulis parameter terbaik langsung ke settings.json setelah optimasi",
    )

    args = parser.parse_args()

    # ── Terapkan override CLI ke module globals ───────────────────────
    if args.symbol:
        OPT_SYMBOL = args.symbol
    if args.timeframe:
        OPT_TIMEFRAME = args.timeframe
    if args.limit:
        OPT_LIMIT = args.limit

    run_optimizer(auto_apply=args.auto_apply)

