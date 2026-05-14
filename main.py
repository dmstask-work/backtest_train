"""
Entry Point Utama — Sistem Backtesting Adaptif (Dynamic Strategy Switching).

Mengorkestrasikan seluruh pipeline dari pengambilan data hingga laporan akhir:

  Fase 1 → DataFetcher     : Ambil OHLCV historis dari exchange (ccxt)
  Fase 2 → IndicatorCalc   : Hitung semua indikator teknikal (vectorized)
  Fase 2 → RegimeFilter    : Klasifikasi kondisi pasar (TRENDING/SIDEWAYS)
  Fase 3 → SignalGenerator : Hasilkan sinyal entry adaptif
  Fase 4 → RiskManager     : Kalkulasi SL, TP, ukuran posisi
  Fase 5 → BacktestEngine  : Simulasi eksekusi & catat semua trade
  Fase 5 → Reporter        : Cetak laporan teks + JSON ke terminal

Cara Menjalankan:
  python main.py

Untuk mengubah parameter (symbol, timeframe, modal, dll),
edit file config.py — tidak perlu mengubah kode di sini.
"""

import logging
import sys

from config import BACKTEST_CONFIG, EXCHANGE_MODE
from backtest_engine import BacktestEngine
from data_fetcher import DataFetcher
from indicators import IndicatorCalculator
from regime_filter import RegimeFilter
from reporter import PerformanceReporter
from risk_manager import RiskManager
from signal_generator import SignalGenerator

# ============================================================
# Konfigurasi Logging Global
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================
# Fungsi Pipeline Utama
# ============================================================
def run_backtest(
    symbol: str        = BACKTEST_CONFIG["symbol"],
    timeframe: str     = BACKTEST_CONFIG["timeframe"],
    limit: int         = BACKTEST_CONFIG["limit"],
    mode: str          = EXCHANGE_MODE,
    show_trade_list: bool = True,
) -> dict:
    """
    Menjalankan siklus backtest lengkap dari fetch data hingga laporan akhir.

    Args:
        symbol:          Simbol pasangan trading (contoh: 'BTC/USDT').
        timeframe:       Timeframe candle (contoh: '1h', '4h', '1d').
        limit:           Jumlah candle historis yang diambil.
        mode:            Mode exchange: 'sandbox' atau 'live'.
        show_trade_list: Tampilkan daftar trade detail jika True.

    Returns:
        Dictionary metrik performa lengkap (juga sudah dicetak ke terminal).
    """
    logger.info("=" * 60)
    logger.info("  BACKTEST ADAPTIF — DYNAMIC STRATEGY SWITCHING")
    logger.info("  Symbol: %s | TF: %s | Candle: %d | Mode: %s",
                symbol, timeframe, limit, mode.upper())
    logger.info("=" * 60)

    try:
        # ── FASE 1: Data Fetcher ─────────────────────────────────────
        logger.info("[FASE 1] Mengambil data OHLCV ...")
        fetcher = DataFetcher(mode=mode)
        df = fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

        # ── FASE 2: Kalkulasi Indikator (vectorized) ─────────────────
        logger.info("[FASE 2] Menghitung indikator teknikal ...")
        calculator = IndicatorCalculator()
        df = calculator.calculate_all(df)

        # ── FASE 2: Market Regime Filter ─────────────────────────────
        logger.info("[FASE 2] Menerapkan Market Regime Filter (ADX) ...")
        regime_filter = RegimeFilter()
        df = regime_filter.classify(df)

        # ── FASE 3: Signal Generation ─────────────────────────────────
        logger.info("[FASE 3] Menghasilkan sinyal trading adaptif ...")
        signal_gen = SignalGenerator()
        df = signal_gen.generate(df)

        # ── FASE 4 & 5: Backtesting Engine ───────────────────────────
        logger.info("[FASE 4/5] Menjalankan mesin backtesting ...")
        risk_manager = RiskManager()
        engine = BacktestEngine(
            initial_capital  = BACKTEST_CONFIG["initial_capital"],
            trade_allocation = BACKTEST_CONFIG["trade_allocation"],
            fee_rate         = BACKTEST_CONFIG["fee_rate"],
            risk_manager     = risk_manager,
        )
        results = engine.run(df)

        # ── FASE 5: Laporan Performa ──────────────────────────────────
        logger.info("[FASE 5] Menghasilkan laporan performa ...")
        reporter = PerformanceReporter(
            initial_capital=BACKTEST_CONFIG["initial_capital"]
        )
        metrics = reporter.calculate_metrics(
            trades       = results["trades"],
            equity_curve = results["equity_curve"],
            final_capital= results["final_capital"],
        )
        reporter.print_report(metrics, symbol=symbol, timeframe=timeframe, mode=mode)

        if show_trade_list and results["trades"]:
            reporter.print_trade_list(results["trades"], max_rows=25)

        logger.info("Backtest selesai dengan sukses.")
        return metrics

    except KeyboardInterrupt:
        logger.info("Backtest dihentikan oleh pengguna.")
        sys.exit(0)

    except Exception as exc:
        logger.exception("Terjadi kesalahan fatal: %s", exc)
        sys.exit(1)


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    run_backtest(
        symbol          = BACKTEST_CONFIG["symbol"],
        timeframe       = BACKTEST_CONFIG["timeframe"],
        limit           = BACKTEST_CONFIG["limit"],
        mode            = EXCHANGE_MODE,
        show_trade_list = True,
    )
