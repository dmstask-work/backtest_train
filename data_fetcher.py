"""
Modul Pengambil Data OHLCV (Fase 1).

Mengambil data candlestick historis dari exchange menggunakan library ccxt.
Mendukung switch antara mode Sandbox/Testnet dan Live Account.
"""

import logging
import ccxt
import pandas as pd

from config import EXCHANGE_CONFIG, EXCHANGE_MODE

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Kelas untuk mengambil data OHLCV historis dari exchange.

    Mendukung mode koneksi sandbox (testnet) dan live secara transparan.
    Seluruh konfigurasi diambil dari config.py sehingga perpindahan mode
    cukup dilakukan satu kali di file config.
    """

    def __init__(self, mode: str = EXCHANGE_MODE) -> None:
        """
        Inisialisasi koneksi ke exchange berdasarkan mode yang dipilih.

        Args:
            mode: Mode koneksi, pilihan: 'sandbox' atau 'live'.
        """
        if mode not in EXCHANGE_CONFIG:
            raise ValueError(
                f"Mode '{mode}' tidak valid. Gunakan 'sandbox' atau 'live'."
            )

        self.mode = mode
        self.cfg = EXCHANGE_CONFIG[mode]
        self.exchange: ccxt.Exchange = self._init_exchange()

    # ------------------------------------------------------------------
    # Inisialisasi Exchange
    # ------------------------------------------------------------------
    def _init_exchange(self) -> ccxt.Exchange:
        """
        Membangun instance ccxt Exchange sesuai konfigurasi.

        Returns:
            Instance ccxt Exchange yang sudah siap digunakan.

        Raises:
            AttributeError: Jika exchange_id tidak dikenali oleh ccxt.
        """
        exchange_id: str = self.cfg["exchange_id"]

        if not hasattr(ccxt, exchange_id):
            raise AttributeError(
                f"Exchange '{exchange_id}' tidak ditemukan di library ccxt."
            )

        exchange_class = getattr(ccxt, exchange_id)
        exchange: ccxt.Exchange = exchange_class(
            {
                "apiKey": self.cfg.get("apiKey", ""),
                "secret": self.cfg.get("secret", ""),
                "options": self.cfg.get("options", {}),
                # Aktifkan rate limiter bawaan ccxt untuk menghindari ban IP
                "enableRateLimit": True,
            }
        )

        if self.cfg.get("sandbox", False):
            exchange.set_sandbox_mode(True)
            logger.info(
                "[DataFetcher] Mode SANDBOX aktif → Exchange: %s", exchange_id
            )
        else:
            logger.info(
                "[DataFetcher] Mode LIVE aktif → Exchange: %s", exchange_id
            )

        return exchange

    # ------------------------------------------------------------------
    # Fungsi Pengambil Data Publik
    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        """
        Mengambil data OHLCV historis dari exchange.

        Args:
            symbol:    Simbol pasangan trading, contoh: 'BTC/USDT'.
            timeframe: Timeframe candle, contoh: '1h', '4h', '1d'.
            limit:     Jumlah candle yang diambil (maks bergantung exchange).

        Returns:
            DataFrame dengan index timestamp dan kolom:
            open, high, low, close, volume.

        Raises:
            ccxt.NetworkError: Saat terjadi kesalahan jaringan.
            ccxt.ExchangeError: Saat terjadi kesalahan dari sisi exchange.
        """
        logger.info(
            "[DataFetcher] Mengambil %d candle | %s [%s] dari %s ...",
            limit,
            symbol,
            timeframe,
            self.cfg["exchange_id"],
        )

        try:
            raw: list = self.exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )
        except ccxt.NetworkError as exc:
            logger.error("[DataFetcher] Kesalahan jaringan: %s", exc)
            raise
        except ccxt.ExchangeError as exc:
            logger.error("[DataFetcher] Kesalahan exchange: %s", exc)
            raise

        if not raw:
            raise ValueError(
                f"Exchange mengembalikan data kosong untuk {symbol} [{timeframe}]."
            )

        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Konversi timestamp milidetik → datetime dan jadikan index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Pastikan semua kolom numerik (vectorized cast)
        df = df.astype(float)

        # Hapus duplikat index jika ada (edge case pada beberapa exchange)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)

        logger.info(
            "[DataFetcher] Berhasil: %d baris | %s → %s",
            len(df),
            df.index[0],
            df.index[-1],
        )

        return df
