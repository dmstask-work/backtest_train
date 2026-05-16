"""
live_bot.py — Stateless Live Execution Node (Phase 2)

Script ini dirancang untuk dijalankan oleh Cron Job / Task Scheduler
setiap pergantian timeframe (mis. setiap 15 menit). Setiap run bersifat
STATELESS: script membaca kondisi exchange secara langsung, membuat
satu keputusan, dan keluar. State disimpan di exchange, bukan di file lokal.

Alur Eksekusi:
  Step 1 → Fetch N candle terakhir langsung dari exchange (bukan cache Parquet)
  Step 2 → Hitung indikator, regime, dan sinyal dari candle yang sudah close
  Step 3 → Sinkronisasi posisi aktif dari exchange
  Step 4 → Eksekusi keputusan: Open / Close / Hold

Keamanan:
  - API key dibaca HANYA dari environment variable, tidak pernah di-hardcode
  - Semua parameter mengalir dari settings.json via config.py
  - Setiap interaksi CCXT dibungkus dalam try-except
  - Crash → log detail + sys.exit(1) → Cron mencoba lagi di jadwal berikutnya

Setup Environment (wajib sebelum menjalankan):
  Linux/macOS : export BINANCE_API_KEY="..." && export BINANCE_API_SECRET="..."
  Windows     : $env:BINANCE_API_KEY="..." ; $env:BINANCE_API_SECRET="..."
  Cron        : Tambahkan variabel ke /etc/environment atau file .env terpisah

Mode Paper Trading (Dry Run):
  python live_bot.py --dry-run
  → Data OHLCV diambil dari market ASLI (public endpoint, tanpa API key)
  → fetch_balance  di-mock → 1000 USDT virtual
  → fetch_positions di-mock → None (tidak ada posisi aktif)
  → create_order DICEGAT — tidak ada order yang dikirim ke exchange
  → Semua keputusan dan sizing dihitung dan dilog secara realistis
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import ccxt
import pandas as pd

from config import BACKTEST_CONFIG, INDICATOR_CONFIG, RISK_CONFIG
from indicators import IndicatorCalculator
from regime_filter import RegimeFilter
from risk_manager import RiskManager
from signal_generator import SIGNAL_BUY, SIGNAL_SELL, SignalGenerator

# =============================================================================
# Logging — File + Console, dikonfigurasi sekali di level modul
# =============================================================================
Path("logs").mkdir(exist_ok=True)

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level   = logging.INFO,
    format  = _LOG_FMT,
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/live_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("live_bot")

# =============================================================================
# Constants — Semua mengalir dari settings.json via config.py, nol hardcode
# =============================================================================
_SYMBOL:          str   = BACKTEST_CONFIG["symbol"]
_TIMEFRAME:       str   = BACKTEST_CONFIG["timeframe"]
_ALLOCATION_PCT:  float = BACKTEST_CONFIG["trade_allocation"]
_FEE_RATE:        float = BACKTEST_CONFIG["fee_rate"]

# 200 candle cukup untuk warm-up EMA-100 + ATR-14 setelah dropna indicator.
# Angka ini independen dari BACKTEST_CONFIG["limit"] yang dipakai untuk backtest.
_CANDLE_LIMIT: int = 200

# Modal virtual yang digunakan saat --dry-run aktif.
_DRY_RUN_CAPITAL: float = 1000.0

# Baca saklar strategi dari settings.json (tidak diekspos oleh config.py)
_SETTINGS_PATH = Path(__file__).parent / "settings.json"
_settings_raw: dict = {}
try:
    if _SETTINGS_PATH.exists():
        _settings_raw = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    logger.warning("[Config] settings.json tidak valid; menggunakan strategi default.")
_stg = _settings_raw.get("strategies", {})
_ENABLE_TREND:    bool = bool(_stg.get("enable_trend_following", True))
_ENABLE_MEAN_REV: bool = bool(_stg.get("enable_mean_reversion",  True))


# =============================================================================
# Exchange Initialization
# =============================================================================
def _init_exchange(dry_run: bool = False) -> ccxt.binance:
    """
    Inisialisasi koneksi ke Binance USDM Futures (Perpetual Contracts).

    Menggunakan ccxt.binance dengan defaultType='future' agar semua operasi
    (fetch_ohlcv, fetch_positions, create_order) bekerja di konteks futures.
    API credentials dibaca eksklusif dari environment variable.

    Saat dry_run=True, API key tidak diwajibkan karena fetch_ohlcv adalah
    endpoint publik yang tidak memerlukan autentikasi.

    Args:
        dry_run: Jika True, lewati validasi API key (mode paper trading).

    Returns:
        Instance ccxt.binance yang dikonfigurasi untuk USDM Futures.

    Raises:
        SystemExit: Jika bukan dry_run dan API key tidak ada di environment.
    """
    api_key    = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not dry_run and (not api_key or not api_secret):
        logger.error(
            "[Exchange] BINANCE_API_KEY atau BINANCE_API_SECRET tidak ditemukan "
            "di environment variable. Set kedua variabel tersebut sebelum "
            "menjalankan live_bot.py."
        )
        sys.exit(1)

    exchange = ccxt.binance({
        "apiKey":          api_key,
        "secret":          api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",  # USDM Perpetual Futures — market asli
        },
    })

    logger.info(
        "[Exchange] Binance USDM Futures diinisialisasi | symbol=%s | tf=%s | dry_run=%s",
        _SYMBOL, _TIMEFRAME, dry_run,
    )
    return exchange


# =============================================================================
# Step 1 — Fetch Live OHLCV
# =============================================================================
def fetch_live_ohlcv(
    exchange:  ccxt.Exchange,
    symbol:    str,
    timeframe: str,
    limit:     int,
) -> pd.DataFrame:
    """
    Mengambil data OHLCV terbaru langsung dari exchange, bukan dari cache Parquet.

    Menggunakan data langsung (bukan cache) untuk memastikan candle terbaru
    tercermin dalam keputusan live trading.

    Args:
        exchange:  Instance ccxt yang sudah diinisialisasi.
        symbol:    Simbol pair (mis. 'BTC/USDT').
        timeframe: Timeframe candle (mis. '15m', '1h', '4h').
        limit:     Jumlah candle yang diambil dari exchange.

    Returns:
        DataFrame OHLCV dengan DatetimeIndex UTC tz-naive, bernama 'timestamp'.

    Raises:
        ccxt.NetworkError:  Masalah koneksi; diteruskan ke caller.
        ccxt.ExchangeError: Error dari exchange; diteruskan ke caller.
        ValueError:         Exchange mengembalikan data kosong.
    """
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw:
        raise ValueError(
            f"Exchange mengembalikan data kosong untuk {symbol} / {timeframe}."
        )

    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # tz-naive UTC
    df.set_index("timestamp", inplace=True)

    logger.info(
        "[Step 1] %d candle diterima: %s → %s",
        len(df), df.index[0], df.index[-1],
    )
    return df


# =============================================================================
# Step 2 — Indicator → Regime → Signal Pipeline
# =============================================================================
def compute_signal(df: pd.DataFrame) -> dict:
    """
    Menjalankan pipeline indikator → regime → sinyal secara penuh.

    Menggunakan baris index -2 (candle yang sudah fully-closed) sebagai
    sumber sinyal, bukan index -1 (candle yang sedang terbentuk / live),
    untuk menghindari sinyal prematur dari candle yang belum close.

    Args:
        df: DataFrame OHLCV mentah dari fetch_live_ohlcv().

    Returns:
        Dictionary dengan key:
          signal   (int):   SIGNAL_BUY (+1), SIGNAL_SELL (-1), atau HOLD (0)
          strategy (str):   Label sub-strategi aktif (mis. 'TREND_FOLLOWING_LONG')
          atr      (float): Nilai ATR candle sinyal, untuk kalkulasi SL/TP
          close    (float): Harga close candle sinyal, estimasi harga entry

    Raises:
        ValueError: Jika DataFrame terlalu pendek setelah dropna indikator.
    """
    indicator_calc = IndicatorCalculator(config=INDICATOR_CONFIG)
    regime_filter  = RegimeFilter()
    signal_gen     = SignalGenerator(
        config=INDICATOR_CONFIG,
        enable_trend_following=_ENABLE_TREND,
        enable_mean_reversion=_ENABLE_MEAN_REV,
    )

    df = indicator_calc.calculate_all(df)
    df = regime_filter.classify(df)
    df = signal_gen.generate(df)

    # Butuh minimal 2 baris: -1 (live) dan -2 (last closed)
    if len(df) < 2:
        raise ValueError(
            "DataFrame terlalu pendek setelah kalkulasi indikator. "
            "Tambah _CANDLE_LIMIT atau periksa parameter indikator."
        )

    # Ambil candle terakhir yang sudah close
    last_closed = df.iloc[-2]

    result = {
        "signal":   int(last_closed["signal"]),
        "strategy": str(last_closed["strategy_used"]),
        "atr":      float(last_closed["atr"]),
        "close":    float(last_closed["close"]),
    }

    logger.info(
        "[Step 2] Sinyal dari candle %s | signal=%+d | strategy=%s | "
        "close=%.4f | atr=%.6f",
        df.index[-2], result["signal"], result["strategy"],
        result["close"], result["atr"],
    )
    return result


# =============================================================================
# Step 3 — Fetch Active Position from Exchange
# =============================================================================
def get_active_position(
    exchange: ccxt.Exchange,
    symbol:   str,
    dry_run:  bool = False,
) -> dict | None:
    """
    Mengambil posisi aktif saat ini dari exchange untuk symbol yang diberikan.

    Posisi dianggap aktif jika jumlah kontrak (contracts) tidak nol.
    Pada Binance USDM Futures, 'contracts' adalah jumlah unit aset, dan
    'side' adalah 'long' atau 'short'.

    Saat dry_run=True, selalu me-return None (posisi virtual = tidak ada)
    tanpa menyentuh endpoint privat exchange.

    Args:
        exchange: Instance ccxt Binance (defaultType='future').
        symbol:   Simbol pair (mis. 'BTC/USDT').
        dry_run:  Jika True, mock posisi sebagai None.

    Returns:
        Dictionary posisi ccxt jika ada posisi aktif, atau None.
        Key penting: 'side' ('long'/'short'), 'contracts' (float).

    Raises:
        ccxt.NetworkError:  Diteruskan ke caller.
        ccxt.ExchangeError: Diteruskan ke caller.
    """
    if dry_run:
        logger.info(
            "[DRY RUN] Posisi virtual = None — fetch_positions dilewati."
        )
        return None

    positions = exchange.fetch_positions([symbol])
    for pos in positions:
        contracts = float(pos.get("contracts") or 0)
        if contracts != 0:
            logger.info(
                "[Step 3] Posisi aktif: side=%s | contracts=%.6f | "
                "entryPrice=%.4f | unrealizedPnl=%.4f",
                pos.get("side"), contracts,
                float(pos.get("entryPrice") or 0),
                float(pos.get("unrealizedPnl") or 0),
            )
            return pos

    logger.info("[Step 3] Tidak ada posisi aktif untuk %s.", symbol)
    return None


# =============================================================================
# Step 4 Helpers — Order Execution
# =============================================================================
def _cancel_open_orders(
    exchange: ccxt.Exchange,
    symbol:   str,
    dry_run:  bool = False,
) -> None:
    """
    Membatalkan semua open order untuk symbol ini.

    Dipanggil sebelum entry atau close posisi untuk mencegah order SL/TP
    orphan yang dapat men-trigger posisi baru secara tidak disengaja.
    Kegagalan cancel dicatat sebagai WARNING saja (non-fatal) agar
    eksekusi order utama tetap berjalan.

    Saat dry_run=True, cancel dilewati sepenuhnya (tidak ada order asli).
    """
    if dry_run:
        logger.info(
            "[DRY RUN] Eksekusi tertahan: Simulasi cancel orders untuk %s.", symbol
        )
        return

    try:
        exchange.cancel_all_orders(symbol)
        logger.info("[Orders] Semua open order untuk %s dibatalkan.", symbol)
    except ccxt.BaseError as exc:
        # Non-fatal: SL/TP aktif dari posisi sebelumnya akan di-handle oleh
        # exchange saat posisi ditutup via reduceOnly.
        logger.warning(
            "[Orders] Gagal cancel open orders untuk %s: %s. "
            "Melanjutkan eksekusi...", symbol, exc,
        )


def _open_position(
    exchange:    ccxt.Exchange,
    symbol:      str,
    direction:   str,
    entry_price: float,
    atr:         float,
    dry_run:     bool = False,
) -> None:
    """
    Membuka posisi baru dengan Market Order + SL + TP di Binance USDM Futures.

    Alur lengkap:
      1. Ambil free balance USDT dari exchange (atau mock 1000 USDT saat dry_run)
      2. Hitung quantity via RiskManager berdasarkan balance
      3. Hitung SL / TP via RiskManager berdasarkan ATR
      4. Kirim MARKET order entry           ← DICEGAT jika dry_run=True
      5. Kirim STOP_MARKET order (SL)       ← DICEGAT jika dry_run=True
      6. Kirim TAKE_PROFIT_MARKET order (TP) ← DICEGAT jika dry_run=True

    Args:
        exchange:    Instance ccxt.
        symbol:      Simbol pair.
        direction:   'LONG' atau 'SHORT'.
        entry_price: Harga close candle terakhir; estimasi harga entry.
        atr:         Nilai ATR candle sinyal.
        dry_run:     Jika True, hitung sizing secara realistis lalu return
                     sebelum create_order dipanggil.

    Raises:
        ccxt.NetworkError, ccxt.ExchangeError: Diteruskan ke caller.
    """
    # ── Dry Run guard: mock balance, hitung sizing, log, cegat sebelum order ─
    if dry_run:
        rm_dry     = RiskManager()
        size_dry   = rm_dry.calculate_position_size(
            capital        = _DRY_RUN_CAPITAL,
            allocation_pct = _ALLOCATION_PCT,
            entry_price    = entry_price,
            fee_rate       = _FEE_RATE,
        )
        levels_dry = rm_dry.calculate_levels(entry_price, atr, direction)
        logger.info(
            "[DRY RUN] Eksekusi tertahan: Simulasi OPEN %s | "
            "modal_virtual=%.2f USDT | allocated=%.2f USDT | qty=%.6f | "
            "entry≈%.4f | SL=%.4f | TP=%.4f | R:R=%.2f",
            direction, _DRY_RUN_CAPITAL, size_dry["trade_capital"],
            size_dry["quantity"], entry_price,
            levels_dry["stop_loss"], levels_dry["take_profit"],
            levels_dry["risk_reward_ratio"],
        )
        return

    # ── Ambil free balance USDT secara real-time ───────────────────────
    balance   = exchange.fetch_balance()
    usdt_free = float((balance.get("USDT") or {}).get("free") or 0)

    if usdt_free <= 0:
        logger.warning(
            "[Execution] Free balance USDT = %.2f. Order dibatalkan.", usdt_free
        )
        return

    # ── Hitung ukuran posisi & level SL/TP ────────────────────────────
    rm     = RiskManager()
    size   = rm.calculate_position_size(
        capital        = usdt_free,
        allocation_pct = _ALLOCATION_PCT,
        entry_price    = entry_price,
        fee_rate       = _FEE_RATE,
    )
    levels = rm.calculate_levels(entry_price, atr, direction)

    qty         = size["quantity"]
    stop_loss   = levels["stop_loss"]
    take_profit = levels["take_profit"]
    entry_side  = "buy"  if direction == "LONG" else "sell"
    close_side  = "sell" if direction == "LONG" else "buy"

    logger.info(
        "[Execution] OPEN %s | balance=%.2f USDT | allocated=%.2f USDT | "
        "qty=%.6f | entry≈%.4f | SL=%.4f | TP=%.4f | R:R=%.2f",
        direction, usdt_free, size["trade_capital"],
        qty, entry_price, stop_loss, take_profit, levels["risk_reward_ratio"],
    )

    # ── 1. Market Entry Order ──────────────────────────────────────────
    entry_order = exchange.create_order(
        symbol = symbol,
        type   = "MARKET",
        side   = entry_side,
        amount = qty,
    )
    logger.info(
        "[Execution] MARKET %s tereksekusi | order_id=%s",
        direction, entry_order.get("id"),
    )

    # ── 2. Stop Loss — STOP_MARKET reduceOnly ─────────────────────────
    sl_order = exchange.create_order(
        symbol = symbol,
        type   = "STOP_MARKET",
        side   = close_side,
        amount = qty,
        price  = None,
        params = {"stopPrice": stop_loss, "reduceOnly": True},
    )
    logger.info(
        "[Execution] SL terpasang @ %.4f | order_id=%s",
        stop_loss, sl_order.get("id"),
    )

    # ── 3. Take Profit — TAKE_PROFIT_MARKET reduceOnly ─────────────────
    tp_order = exchange.create_order(
        symbol = symbol,
        type   = "TAKE_PROFIT_MARKET",
        side   = close_side,
        amount = qty,
        price  = None,
        params = {"stopPrice": take_profit, "reduceOnly": True},
    )
    logger.info(
        "[Execution] TP terpasang @ %.4f | order_id=%s",
        take_profit, tp_order.get("id"),
    )


def _close_position(
    exchange: ccxt.Exchange,
    symbol:   str,
    position: dict,
    dry_run:  bool = False,
) -> None:
    """
    Menutup posisi aktif dengan Market Order berlawanan arah.

    Alur:
      1. Cancel semua open order (SL/TP yang terpasang pada posisi ini)
      2. Kirim Market Order berlawanan dengan reduceOnly=True

    Saat dry_run=True, hanya mencatat ke log dan return sebelum order dikirim.

    Args:
        exchange: Instance ccxt.
        symbol:   Simbol pair.
        position: Dictionary posisi dari get_active_position().
        dry_run:  Jika True, cegat sebelum create_order dipanggil.

    Raises:
        ccxt.NetworkError, ccxt.ExchangeError: Diteruskan ke caller.
    """
    pos_side  = str(position.get("side", "long"))
    contracts = float(position.get("contracts") or 0)

    if dry_run:
        logger.info(
            "[DRY RUN] Eksekusi tertahan: Simulasi CLOSE %s | contracts=%.6f",
            pos_side.upper(), contracts,
        )
        return

    close_side = "sell" if pos_side == "long" else "buy"

    _cancel_open_orders(exchange, symbol)

    close_order = exchange.create_order(
        symbol = symbol,
        type   = "MARKET",
        side   = close_side,
        amount = contracts,
        params = {"reduceOnly": True},
    )
    logger.info(
        "[Execution] Posisi %s DITUTUP | contracts=%.6f | order_id=%s",
        pos_side.upper(), contracts, close_order.get("id"),
    )


# =============================================================================
# Main Execution Logic
# =============================================================================
def run(dry_run: bool = False) -> None:
    """
    Entry point utama — satu siklus keputusan stateless.

    Dapat dipanggil langsung oleh __main__ atau oleh scheduler eksternal.
    Setiap pemanggilan membaca kondisi exchange dari nol tanpa asumsi state.

    Args:
        dry_run: Jika True, semua create_order dicegat (Paper Trading mode).

    Exit codes:
      0 → Siklus selesai normal (Hold, Open, atau Close berhasil / disimulasikan)
      1 → Error fatal yang memerlukan investigasi (network, auth, data)
    """
    logger.info("=" * 65)
    logger.info(
        "[LiveBot] === SIKLUS BARU | symbol=%s | timeframe=%s ===",
        _SYMBOL, _TIMEFRAME,
    )
    logger.info(
        "[LiveBot] Strategi aktif: Trend=%s | MeanRev=%s",
        "ON" if _ENABLE_TREND else "OFF",
        "ON" if _ENABLE_MEAN_REV else "OFF",
    )
    if dry_run:
        logger.info(
            "[LiveBot] *** MODE DRY RUN (Paper Trading) *** "
            "— create_order TIDAK akan terpanggil. Modal virtual = %.2f USDT",
            _DRY_RUN_CAPITAL,
        )
    logger.info("=" * 65)

    # ── Step 1: Inisialisasi Exchange ──────────────────────────────────
    # SystemExit dari _init_exchange tidak ditangkap; env var missing = halt.
    exchange = _init_exchange(dry_run=dry_run)

    # ── Step 2: Fetch OHLCV ───────────────────────────────────────────
    try:
        df_raw = fetch_live_ohlcv(exchange, _SYMBOL, _TIMEFRAME, _CANDLE_LIMIT)
    except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
        logger.error("[Step 1] Network/Exchange error saat fetch OHLCV: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("[Step 1] Data tidak valid dari exchange: %s", exc)
        sys.exit(1)

    # ── Step 3: Hitung Sinyal ─────────────────────────────────────────
    try:
        sig = compute_signal(df_raw)
    except Exception as exc:
        logger.error("[Step 2] Gagal menghitung sinyal: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Step 4: Sinkronisasi Posisi ───────────────────────────────────
    try:
        active_pos = get_active_position(exchange, _SYMBOL, dry_run=dry_run)
    except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
        logger.error("[Step 3] Gagal sinkronisasi posisi dari exchange: %s", exc)
        sys.exit(1)

    # ── Step 5: Decision & Execution ──────────────────────────────────
    try:
        if active_pos is None:
            # ── Tidak ada posisi: buka jika ada sinyal ─────────────────
            if sig["signal"] == SIGNAL_BUY:
                logger.info(
                    "[Decision] OPEN LONG | strategy=%s", sig["strategy"]
                )
                _cancel_open_orders(exchange, _SYMBOL, dry_run=dry_run)
                _open_position(
                    exchange, _SYMBOL, "LONG", sig["close"], sig["atr"],
                    dry_run=dry_run,
                )

            elif sig["signal"] == SIGNAL_SELL:
                logger.info(
                    "[Decision] OPEN SHORT | strategy=%s", sig["strategy"]
                )
                _cancel_open_orders(exchange, _SYMBOL, dry_run=dry_run)
                _open_position(
                    exchange, _SYMBOL, "SHORT", sig["close"], sig["atr"],
                    dry_run=dry_run,
                )

            else:
                logger.info(
                    "[Decision] HOLD — tidak ada sinyal entry. "
                    "strategy=%s", sig["strategy"],
                )

        else:
            pos_side = str(active_pos.get("side", "long"))  # 'long' atau 'short'

            # Deteksi sinyal berlawanan dengan posisi aktif
            opposing_signal = (
                (pos_side == "long"  and sig["signal"] == SIGNAL_SELL) or
                (pos_side == "short" and sig["signal"] == SIGNAL_BUY)
            )

            if opposing_signal:
                logger.info(
                    "[Decision] CLOSE %s — sinyal berlawanan (%s signal=%+d)",
                    pos_side.upper(), sig["strategy"], sig["signal"],
                )
                _close_position(exchange, _SYMBOL, active_pos, dry_run=dry_run)

            else:
                # Arah sama atau sinyal HOLD → pertahankan posisi
                logger.info(
                    "[Decision] HOLD — posisi %s aktif, sinyal=%+d "
                    "(tidak berlawanan). SL/TP exchange yang akan menutup.",
                    pos_side.upper(), sig["signal"],
                )

    except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
        logger.error("[Step 4] Gagal eksekusi order: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("[LiveBot] Siklus selesai dengan normal.\n")


# =============================================================================
# CLI Argument Parser
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live_bot.py",
        description="Stateless Live Execution Node — Phase 2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help=(
            "Aktifkan mode Paper Trading. Data OHLCV diambil dari market asli, "
            "namun semua create_order DICEGAT dan tidak dikirim ke exchange. "
            "fetch_balance di-mock ke %.2f USDT virtual. "
            "fetch_positions di-mock ke None." % _DRY_RUN_CAPITAL
        ),
    )
    return parser


# =============================================================================
if __name__ == "__main__":
    _args = _build_arg_parser().parse_args()
    run(dry_run=_args.dry_run)
