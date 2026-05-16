"""
Entry Point Utama — Sistem Backtesting Adaptif v2 (Dynamic Strategy Switching).

Mengorkestrasikan seluruh pipeline dari pengambilan data hingga laporan akhir:

  Fase 1 → DataFetcher     : Ambil OHLCV historis dari exchange (ccxt)
  Fase 2 → IndicatorCalc   : Hitung semua indikator teknikal (vectorized)
  Fase 2 → RegimeFilter    : Klasifikasi kondisi pasar (TRENDING/SIDEWAYS)
  Fase 3 → SignalGenerator : Hasilkan sinyal entry adaptif
  Fase 4 → RiskManager     : Kalkulasi SL, TP, ukuran posisi
  Fase 5 → BacktestEngine  : Simulasi eksekusi & catat semua trade
  Fase 5 → Reporter        : Cetak laporan teks + JSON ke terminal

Cara Menjalankan:
  python main.py                              # Gunakan settings.json default
  python main.py --symbol ETH/USDT --tf 4h   # Override symbol & timeframe
  python main.py --alloc 0.4 --capital 5000  # Override modal & alokasi
  python main.py --no-meanrev                # Hanya jalankan Trend-Following
  python main.py --profile ETH               # Load profil preset dari settings.json
  python main.py --settings custom.json      # Gunakan file settings lain
  python main.py --log-level DEBUG           # Mode verbose untuk debugging

Semua parameter default berasal dari settings.json.
Argumen CLI selalu mengalahkan nilai di settings.json (override).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from config import BACKTEST_CONFIG, EXCHANGE_MODE, INDICATOR_CONFIG, RISK_CONFIG
from backtest_engine import BacktestEngine
from data_fetcher import DataFetcher
from indicators import IndicatorCalculator
from regime_filter import RegimeFilter
from reporter import PerformanceReporter
from risk_manager import RiskManager
from signal_generator import SignalGenerator

# ============================================================
# Fungsi Loader Settings
# ============================================================
def load_settings(settings_path: str) -> dict:
    """
    Memuat konfigurasi dari file settings.json.

    Jika file tidak ditemukan, fungsi akan mengembalikan dictionary kosong
    sehingga seluruh nilai diambil dari config.py sebagai fallback.

    Args:
        settings_path: Path ke file JSON konfigurasi.

    Returns:
        Dictionary isi settings.json, atau {} jika file tidak ada.
    """
    path = Path(settings_path)
    if not path.exists():
        logging.getLogger(__name__).warning(
            "[Settings] File '%s' tidak ditemukan. Menggunakan config.py sebagai fallback.",
            settings_path,
        )
        return {}

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Hapus semua key yang diawali '_comment' (komentar di JSON)
    def strip_comments(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: strip_comments(v)
                for k, v in obj.items()
                if not k.startswith("_comment")
            }
        return obj

    settings = strip_comments(raw)
    logging.getLogger(__name__).info(
        "[Settings] Konfigurasi dimuat dari '%s'.", settings_path
    )
    return settings


def apply_profile(settings: dict, profile_name: str) -> dict:
    """
    Menimpa nilai indikator dan backtest dari profil preset di settings.json.

    Args:
        settings:     Dictionary settings yang sudah dimuat.
        profile_name: Nama profil, contoh: 'ETH', 'SOL', 'BTC'.

    Returns:
        Settings yang sudah di-overlay dengan nilai profil.
    """
    profiles = settings.get("profiles", {})
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise ValueError(
            f"Profil '{profile_name}' tidak ditemukan di settings.json. "
            f"Profil tersedia: {available}"
        )

    profile = profiles[profile_name]
    # Overlay ke bagian 'indicators' dan 'backtest'
    settings.setdefault("indicators", {}).update(
        {k: v for k, v in profile.items() if k in settings.get("indicators", {})}
    )
    settings.setdefault("backtest", {}).update(
        {k: v for k, v in profile.items() if k in settings.get("backtest", {})}
    )

    logging.getLogger(__name__).info(
        "[Settings] Profil '%s' diterapkan.", profile_name
    )
    return settings


# ============================================================
# Fungsi Pembuat Config Gabungan
# ============================================================
def build_config(settings: dict) -> dict:
    """
    Menggabungkan nilai dari settings.json dengan default dari config.py.

    Prioritas: settings.json > config.py

    Args:
        settings: Dictionary dari load_settings().

    Returns:
        Dictionary config terunifikasi dengan kunci:
          exchange_mode, symbol, timeframe, limit,
          initial_capital, trade_allocation, fee_rate,
          enable_trend_following, enable_mean_reversion,
          indicators (dict), risk (dict)
    """
    bk  = settings.get("backtest",   {})
    ind = settings.get("indicators", {})
    rsk = settings.get("risk",       {})
    stg = settings.get("strategies", {})
    exc = settings.get("exchange",   {})

    # Gabungkan indikator: settings.json menimpa config.py
    merged_indicators = {**INDICATOR_CONFIG, **ind}

    return {
        "exchange_mode":          exc.get("mode",              EXCHANGE_MODE),
        "symbol":                 bk.get("symbol",             BACKTEST_CONFIG["symbol"]),
        "timeframe":              bk.get("timeframe",          BACKTEST_CONFIG["timeframe"]),
        "limit":                  bk.get("limit",              BACKTEST_CONFIG["limit"]),
        "initial_capital":        bk.get("initial_capital",    BACKTEST_CONFIG["initial_capital"]),
        "trade_allocation":       bk.get("trade_allocation",   BACKTEST_CONFIG["trade_allocation"]),
        "fee_rate":               bk.get("fee_rate",           BACKTEST_CONFIG["fee_rate"]),
        "enable_trend_following": stg.get("enable_trend_following", True),
        "enable_mean_reversion":  stg.get("enable_mean_reversion",  True),
        "indicators":             merged_indicators,
        "risk": {
            "sl_multiplier": rsk.get("sl_multiplier", RISK_CONFIG["sl_multiplier"]),
            "tp_multiplier": rsk.get("tp_multiplier", RISK_CONFIG["tp_multiplier"]),
        },
    }


# ============================================================
# CLI Argument Parser
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Membangun parser argumen CLI dengan semua opsi yang tersedia.

    Returns:
        ArgumentParser yang siap digunakan.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Sistem Backtesting Kripto Adaptif (Dynamic Strategy Switching)\n"
            "Semua argumen CLI akan menimpa nilai di settings.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh:\n"
            "  python main.py --symbol ETH/USDT --tf 4h\n"
            "  python main.py --alloc 0.4 --capital 5000 --no-meanrev\n"
            "  python main.py --profile SOL --tf 1h --limit 500\n"
            "  python main.py --settings custom.json --log-level DEBUG\n"
        ),
    )

    # ── Sumber konfigurasi ─────────────────────────────────────────────
    parser.add_argument(
        "--settings", "-c",
        default="settings.json",
        metavar="PATH",
        help="Path ke file settings JSON. (default: settings.json)",
    )
    parser.add_argument(
        "--profile", "-p",
        default=None,
        metavar="NAMA",
        help="Terapkan profil preset dari settings.json (contoh: ETH, SOL, BTC).",
    )

    # ── Parameter backtest ─────────────────────────────────────────────
    parser.add_argument(
        "--symbol", "-s",
        default=None,
        metavar="PAIR",
        help="Simbol pasangan trading (contoh: ETH/USDT, SOL/USDT).",
    )
    parser.add_argument(
        "--tf", "-t",
        default=None,
        dest="timeframe",
        metavar="TF",
        help="Timeframe candle (contoh: 1h, 4h, 1d).",
    )
    parser.add_argument(
        "--limit", "-l",
        default=None,
        type=int,
        metavar="N",
        help="Jumlah candle historis yang diambil (contoh: 500).",
    )
    parser.add_argument(
        "--alloc", "-a",
        default=None,
        type=float,
        dest="trade_allocation",
        metavar="FRAKSI",
        help="Fraksi modal per trade, 0.0–1.0 (contoh: 0.4 untuk 40%%).",
    )
    parser.add_argument(
        "--capital",
        default=None,
        type=float,
        metavar="USD",
        help="Modal awal simulasi dalam USD (contoh: 5000).",
    )
    parser.add_argument(
        "--mode", "-m",
        default=None,
        choices=["sandbox", "live"],
        help="Mode exchange: 'sandbox' (testnet) atau 'live' (akun nyata).",
    )

    # ── Strategy Switcher ─────────────────────────────────────────────
    strat_group = parser.add_argument_group("Saklar Strategi")
    strat_group.add_argument(
        "--no-trend",
        action="store_true",
        help="Nonaktifkan strategi Trend-Following (hanya Mean-Reversion yang berjalan).",
    )
    strat_group.add_argument(
        "--no-meanrev",
        action="store_true",
        help="Nonaktifkan strategi Mean-Reversion (hanya Trend-Following yang berjalan).",
    )

    # ── Output ────────────────────────────────────────────────────────
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--no-trades",
        action="store_true",
        help="Sembunyikan tabel daftar trade detail di output.",
    )
    output_group.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Level detail log (DEBUG|INFO|WARNING|ERROR). (default: INFO)",
    )

    return parser


# ============================================================
# Fungsi Pipeline Utama
# ============================================================
def run_backtest(cfg: dict, show_trade_list: bool = True) -> dict:
    """
    Menjalankan siklus backtest lengkap dari fetch data hingga laporan akhir.

    Args:
        cfg:             Dictionary config terunifikasi dari build_config().
        show_trade_list: Tampilkan tabel trade detail jika True.

    Returns:
        Dictionary metrik performa lengkap.
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 64)
    logger.info("  BACKTEST ADAPTIF v2 — DYNAMIC STRATEGY SWITCHING")
    logger.info(
        "  Symbol: %s | TF: %s | Candle: %d | Mode: %s",
        cfg["symbol"], cfg["timeframe"], cfg["limit"],
        cfg["exchange_mode"].upper(),
    )
    logger.info(
        "  Strategi: Trend-Following=%s | Mean-Reversion=%s",
        "ON" if cfg["enable_trend_following"] else "OFF",
        "ON" if cfg["enable_mean_reversion"]  else "OFF",
    )
    logger.info("=" * 64)

    try:
        # ── FASE 1: Data Fetcher ─────────────────────────────────────
        logger.info("[FASE 1] Mengambil data OHLCV ...")
        fetcher = DataFetcher(mode=cfg["exchange_mode"])
        df = fetcher.fetch_ohlcv(
            symbol=cfg["symbol"],
            timeframe=cfg["timeframe"],
            limit=cfg["limit"],
        )

        # ── FASE 2: Kalkulasi Indikator (vectorized) ─────────────────
        logger.info("[FASE 2] Menghitung indikator teknikal ...")
        calculator = IndicatorCalculator(config=cfg["indicators"])
        df = calculator.calculate_all(df)

        # ── FASE 2: Market Regime Filter ─────────────────────────────
        logger.info("[FASE 2] Menerapkan Market Regime Filter (ADX) ...")
        regime_filter = RegimeFilter(
            adx_threshold=cfg["indicators"]["adx_threshold"]
        )
        df = regime_filter.classify(df)

        # ── FASE 3: Signal Generation ─────────────────────────────────
        logger.info("[FASE 3] Menghasilkan sinyal trading adaptif ...")
        signal_gen = SignalGenerator(
            config=cfg["indicators"],
            enable_trend_following=cfg["enable_trend_following"],
            enable_mean_reversion=cfg["enable_mean_reversion"],
        )
        df = signal_gen.generate(df)

        # ── FASE 4 & 5: Backtesting Engine ───────────────────────────
        logger.info("[FASE 4/5] Menjalankan mesin backtesting ...")
        risk_manager = RiskManager(
            sl_multiplier=cfg["risk"]["sl_multiplier"],
            tp_multiplier=cfg["risk"]["tp_multiplier"],
        )
        engine = BacktestEngine(
            initial_capital  = cfg["initial_capital"],
            trade_allocation = cfg["trade_allocation"],
            fee_rate         = cfg["fee_rate"],
            risk_manager     = risk_manager,
        )
        results = engine.run(df)

        # ── FASE 5: Laporan Performa ──────────────────────────────────
        logger.info("[FASE 5] Menghasilkan laporan performa ...")
        reporter = PerformanceReporter(initial_capital=cfg["initial_capital"])
        metrics = reporter.calculate_metrics(
            trades       = results["trades"],
            equity_curve = results["equity_curve"],
            final_capital= results["final_capital"],
        )
        reporter.print_report(
            metrics,
            symbol    = cfg["symbol"],
            timeframe = cfg["timeframe"],
            mode      = cfg["exchange_mode"],
            export_csv= True,  # CSV Audit trail dihidupkan untuk AI
        )

        if show_trade_list and results["trades"]:
            reporter.print_trade_list(results["trades"], max_rows=25)

        logger.info("Backtest selesai dengan sukses.")
        return metrics

    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Backtest dihentikan oleh pengguna.")
        sys.exit(0)

    except Exception as exc:
        logging.getLogger(__name__).exception("Terjadi kesalahan fatal: %s", exc)
        sys.exit(1)


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    # ── 1. Parse argumen CLI ──────────────────────────────────────────
    parser = build_arg_parser()
    args   = parser.parse_args()

    # ── 2. Setup logging sesuai level dari CLI ────────────────────────
    Path("logs").mkdir(exist_ok=True)
    _log_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _log_lvl = getattr(logging, args.log_level)
    logging.basicConfig(
        level   = _log_lvl,
        format  = _log_fmt,
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/main_execution.log", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)

    # ── 3. Muat settings.json ─────────────────────────────────────────
    settings = load_settings(args.settings)

    # ── 4. Terapkan profil jika --profile diberikan ───────────────────
    if args.profile:
        settings = apply_profile(settings, args.profile)

    # ── 5. Bangun config terpadu dari settings + config.py fallback ───
    cfg = build_config(settings)

    # ── 6. Override dengan argumen CLI (prioritas tertinggi) ──────────
    if args.symbol           is not None: cfg["symbol"]           = args.symbol
    if args.timeframe        is not None: cfg["timeframe"]        = args.timeframe
    if args.limit            is not None: cfg["limit"]            = args.limit
    if args.trade_allocation is not None: cfg["trade_allocation"] = args.trade_allocation
    if args.capital          is not None: cfg["initial_capital"]  = args.capital
    if args.mode             is not None: cfg["exchange_mode"]    = args.mode

    # Strategy switcher: --no-trend / --no-meanrev mematikan strategi
    if args.no_trend:   cfg["enable_trend_following"] = False
    if args.no_meanrev: cfg["enable_mean_reversion"]  = False

    # ── 7. Validasi alokasi modal ─────────────────────────────────────
    if not (0.0 < cfg["trade_allocation"] <= 1.0):
        logger.error(
            "Nilai --alloc harus antara 0.0 dan 1.0. Diterima: %.2f",
            cfg["trade_allocation"],
        )
        sys.exit(1)

    # ── 8. Jalankan backtest ──────────────────────────────────────────
    run_backtest(cfg=cfg, show_trade_list=not args.no_trades)

