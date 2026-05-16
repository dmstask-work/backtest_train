"""
Konfigurasi terpusat untuk sistem backtesting adaptif.

Loader hierarki:
  1. Baca settings.json dari direktori yang sama dengan file ini.
  2. Jika ditemukan, gabungkan (merge) nilai JSON ke dalam defaults.
  3. Jika tidak ditemukan atau parse gagal, gunakan defaults sepenuhnya.

Kredensi live (API key / secret) TIDAK PERNAH disimpan di kode.
Gunakan environment variables: API_KEY dan SECRET_KEY.
"""

import json
import logging
import os
from copy import deepcopy
from pathlib import Path

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ============================================================
# DEFAULT CONFIGS — Sumber kebenaran tunggal untuk semua nilai
# ============================================================

DEFAULT_EXCHANGE_MODE: str = "sandbox"
DEFAULT_EXCHANGE_ID:   str = "binance"

DEFAULT_BACKTEST_CONFIG: dict = {
    "symbol":           "SOL/USDT",
    "timeframe":        "4h",
    "limit":            5000,
    "initial_capital":  1000.0,
    "trade_allocation": 0.20,
    "fee_rate":         0.001,
}

DEFAULT_INDICATOR_CONFIG: dict = {
    # ADX — Market Regime Filter
    "adx_period":                  14,
    "adx_threshold":               25,

    # EMA
    "ema_fast":                    20,
    "ema_slow":                    50,

    # MACD
    "macd_fast":                   12,
    "macd_slow":                   26,
    "macd_signal":                 9,

    # Supertrend
    "supertrend_period":           10,
    "supertrend_multiplier":       3.0,

    # Bollinger Bands
    "bb_period":                   20,
    "bb_std":                      2.0,
    "bb_bandwidth_expansion_factor": 1.2,

    # RSI
    "rsi_period":                  14,
    "rsi_oversold":                30,
    "rsi_overbought":              70,

    # ATR
    "atr_period":                  14,

    # Volume SMA
    "volume_sma_period":           20,
}

DEFAULT_RISK_CONFIG: dict = {
    "sl_multiplier": 1.5,
    "tp_multiplier": 3.0,
}


# ============================================================
# JSON LOADER
# ============================================================

_SETTINGS_PATH = Path(__file__).parent / "settings.json"


def _filter_comments(d: dict) -> dict:
    """Hapus semua key yang diawali '_comment' dari sebuah dict (satu level)."""
    return {k: v for k, v in d.items() if not k.startswith("_comment")}


def _load_settings() -> dict:
    """
    Muat settings.json dan kembalikan sebagai dict.

    Returns:
        Dict berisi konten JSON yang sudah difilter, atau dict kosong
        jika file tidak ditemukan / tidak valid.
    """
    if not _SETTINGS_PATH.exists():
        logger.warning(
            "settings.json tidak ditemukan di '%s'. "
            "Menggunakan konfigurasi default.",
            _SETTINGS_PATH,
        )
        return {}

    try:
        raw = _SETTINGS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        logger.debug("settings.json berhasil dimuat dari '%s'.", _SETTINGS_PATH)
        return data
    except json.JSONDecodeError as exc:
        logger.warning(
            "Gagal mem-parse settings.json (%s). "
            "Menggunakan konfigurasi default.",
            exc,
        )
        return {}


def _merge(default: dict, override: dict) -> dict:
    """
    Gabungkan override ke dalam salinan default.

    Key yang diawali '_comment' dibuang secara otomatis.
    Hanya key yang sudah ada di default yang diterima (whitelist approach)
    untuk mencegah injeksi key asing dari JSON eksternal.

    Args:
        default:  Dict nilai bawaan.
        override: Dict nilai dari JSON (boleh mengandung _comment keys).

    Returns:
        Dict baru hasil penggabungan.
    """
    result = deepcopy(default)
    clean  = _filter_comments(override)
    for key, value in clean.items():
        if key in result:
            result[key] = value
    return result


# ============================================================
# BUILD DYNAMIC CONFIGS
# ============================================================

_settings = _load_settings()

# ── Exchange ──────────────────────────────────────────────────────────────────
_exchange_section = _filter_comments(_settings.get("exchange", {}))
EXCHANGE_MODE: str = str(_exchange_section.get("mode", DEFAULT_EXCHANGE_MODE))
_exchange_id:  str = str(_exchange_section.get("exchange_id", DEFAULT_EXCHANGE_ID))

if EXCHANGE_MODE not in ("sandbox", "live"):
    logger.warning(
        "Nilai 'exchange.mode' tidak valid ('%s'). Jatuh kembali ke 'sandbox'.",
        EXCHANGE_MODE,
    )
    EXCHANGE_MODE = "sandbox"

EXCHANGE_CONFIG: dict = {
    "sandbox": {
        "exchange_id":    _exchange_id,
        "sandbox":        False,
        "enableRateLimit": True,
        "apiKey":         "",
        "secret":         "",
        "options": {
            "defaultType": "spot",
        },
    },
    "live": {
        "exchange_id":    _exchange_id,
        "sandbox":        False,
        "enableRateLimit": True,
        "apiKey":         os.getenv("API_KEY", ""),
        "secret":         os.getenv("SECRET_KEY", ""),
        "options": {
            "defaultType": "spot",
        },
    },
}

# ── Backtest ──────────────────────────────────────────────────────────────────
BACKTEST_CONFIG: dict = _merge(
    DEFAULT_BACKTEST_CONFIG,
    _filter_comments(_settings.get("backtest", {})),
)

# ── Indicators ────────────────────────────────────────────────────────────────
INDICATOR_CONFIG: dict = _merge(
    DEFAULT_INDICATOR_CONFIG,
    _filter_comments(_settings.get("indicators", {})),
)

# ── Risk ──────────────────────────────────────────────────────────────────────
RISK_CONFIG: dict = _merge(
    DEFAULT_RISK_CONFIG,
    _filter_comments(_settings.get("risk", {})),
)

# ── Startup log ───────────────────────────────────────────────────────────────
logger.info(
    "Config dimuat — mode: '%s' | exchange_id: '%s' | sumber: %s",
    EXCHANGE_MODE,
    _exchange_id,
    _SETTINGS_PATH if _settings else "defaults",
)
