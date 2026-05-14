"""
Konfigurasi terpusat untuk sistem backtesting adaptif.

Mendukung switch mudah antara mode Sandbox/Testnet dan Real Account (Live)
hanya dengan mengubah nilai EXCHANGE_MODE.
"""

# ============================================================
# SWITCH UTAMA: Ubah ke 'live' untuk akun nyata
# ============================================================
EXCHANGE_MODE: str = "sandbox"  # pilihan: 'sandbox' | 'live'

# ============================================================
# Konfigurasi Exchange (ccxt)
# ============================================================
EXCHANGE_CONFIG: dict = {
    "sandbox": {
        "exchange_id": "binance",
        "sandbox": True,
        "apiKey": "",         # Isi dengan API key testnet jika diperlukan
        "secret": "",         # Isi dengan secret key testnet jika diperlukan
        "options": {
            "defaultType": "spot",
        },
    },
    "live": {
        "exchange_id": "binance",
        "sandbox": False,
        "apiKey": "YOUR_API_KEY",    # Ganti dengan API key live
        "secret": "YOUR_SECRET_KEY", # Ganti dengan secret key live
        "options": {
            "defaultType": "spot",
        },
    },
}

# ============================================================
# Parameter Backtest
# ============================================================
BACKTEST_CONFIG: dict = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "limit": 1000,              # Jumlah candle historis yang diambil
    "initial_capital": 1000.0,  # Modal awal dalam USD
    "trade_allocation": 0.10,   # Alokasi modal per trade: 10%
    "fee_rate": 0.001,          # Biaya transaksi per sisi: 0.1%
}

# ============================================================
# Parameter Indikator Teknikal
# ============================================================
INDICATOR_CONFIG: dict = {
    # ADX - Market Regime Filter
    "adx_period": 14,
    "adx_threshold": 25,       # ADX > 25 => TRENDING, <= 25 => SIDEWAYS

    # EMA
    "ema_fast": 20,
    "ema_slow": 50,

    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # Bollinger Bands
    "bb_period": 20,
    "bb_std": 2.0,

    # RSI
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,

    # ATR - Risk Management
    "atr_period": 14,
}

# ============================================================
# Parameter Manajemen Risiko (ATR-based)
# ============================================================
RISK_CONFIG: dict = {
    "sl_multiplier": 1.5,  # Stop Loss  = Entry - (1.5 × ATR)
    "tp_multiplier": 3.0,  # Take Profit = Entry + (3.0 × ATR)
}
