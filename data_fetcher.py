"""
Modul Pengambil Data OHLCV (Fase 1).

Mengambil data candlestick historis dari exchange menggunakan library ccxt.
Mendukung switch antara mode Sandbox/Testnet dan Live Account.

v2: Implementasi pagination otomatis untuk melampaui batas 1000 candle
    per-request yang dikenakan oleh Binance dan sebagian besar exchange.
"""

import logging
import time

import ccxt
import pandas as pd

from config import EXCHANGE_CONFIG, EXCHANGE_MODE

logger = logging.getLogger(__name__)

# Jumlah candle maksimum per satu request (batas Binance)
_BATCH_SIZE: int = 1000


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
    # Fungsi Pengambil Data Publik (dengan Pagination)
    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> pd.DataFrame:
        """
        Mengambil data OHLCV historis dengan pagination otomatis.

        Binance (dan sebagian besar exchange) membatasi satu request hanya
        1000 candle. Fungsi ini mengatasi batasan tersebut dengan looping
        bertahap menggunakan parameter 'since' (timestamp mundur) hingga
        jumlah candle yang diminta terpenuhi.

        Mekanisme:
          1. Hitung titik awal historis: now - (limit × tf_duration_ms)
          2. Loop: ambil batch 1000 candle, geser 'since' ke depan
          3. Rate-limit delay di setiap iterasi agar IP tidak diblokir
          4. Setelah semua batch terkumpul: concat → dedup → sort → trim

        Args:
            symbol:    Simbol pasangan trading, contoh: 'BTC/USDT'.
            timeframe: Timeframe candle, contoh: '1h', '4h', '1d'.
            limit:     Total candle yang diinginkan (bisa > 1000).

        Returns:
            DataFrame dengan index timestamp (UTC) dan kolom:
            open, high, low, close, volume. Jumlah baris = min(limit, data tersedia).

        Raises:
            ccxt.NetworkError:  Saat terjadi kesalahan jaringan.
            ccxt.ExchangeError: Saat terjadi kesalahan dari sisi exchange.
            ValueError:         Jika exchange mengembalikan data kosong sama sekali.
        """
        logger.info(
            "[DataFetcher] Memulai pengambilan %d candle | %s [%s] | Exchange: %s",
            limit,
            symbol,
            timeframe,
            self.cfg["exchange_id"],
        )

        # ── Kalkulasi durasi satu candle dalam milidetik ──────────────
        # ccxt.Exchange.parse_timeframe() mengembalikan nilai dalam detik
        tf_seconds: int = self.exchange.parse_timeframe(timeframe)
        tf_ms:      int = tf_seconds * 1000

        # Jeda minimum antar request (gunakan rateLimit bawaan ccxt,
        # minimal 500 ms sebagai safety floor untuk mencegah ban IP)
        rate_delay: float = max(self.exchange.rateLimit / 1000, 0.5)

        # ── Hitung titik awal historis ────────────────────────────────
        # Mundur sejumlah (limit × durasi_candle) dari waktu sekarang
        now_ms:    int = self.exchange.milliseconds()
        since_ms:  int = now_ms - (limit * tf_ms)

        # ── Pagination Loop ───────────────────────────────────────────
        all_raw:    list = []
        batch_num:  int  = 0
        total_batches: int = -(-limit // _BATCH_SIZE)  # ceiling division

        while len(all_raw) < limit:
            remaining   = limit - len(all_raw)
            to_fetch    = min(_BATCH_SIZE, remaining)
            batch_num  += 1

            logger.info(
                "[DataFetcher] Batch %d/%d | Mengambil %d candle sejak %s ...",
                batch_num,
                total_batches,
                to_fetch,
                pd.Timestamp(since_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
            )

            try:
                batch: list = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since_ms,
                    limit=to_fetch,
                )
            except ccxt.NetworkError as exc:
                logger.error(
                    "[DataFetcher] Kesalahan jaringan pada batch %d: %s",
                    batch_num, exc,
                )
                raise
            except ccxt.ExchangeError as exc:
                logger.error(
                    "[DataFetcher] Kesalahan exchange pada batch %d: %s",
                    batch_num, exc,
                )
                raise

            if not batch:
                logger.warning(
                    "[DataFetcher] Batch %d mengembalikan data kosong. "
                    "Data historis mungkin tidak mencukupi untuk limit=%d.",
                    batch_num, limit,
                )
                break

            all_raw.extend(batch)

            # Geser 'since' ke tepat setelah candle terakhir yang diterima
            since_ms = batch[-1][0] + tf_ms

            # Exchange mengembalikan lebih sedikit dari yang diminta →
            # tidak ada lagi data historis di periode ini
            if len(batch) < to_fetch:
                logger.info(
                    "[DataFetcher] Exchange mengembalikan %d < %d candle. "
                    "Batas data historis tercapai.",
                    len(batch), to_fetch,
                )
                break

            # Rate-limit delay — WAJIB jika masih ada batch berikutnya
            if len(all_raw) < limit:
                logger.debug(
                    "[DataFetcher] Rate-limit delay %.2f detik ...", rate_delay
                )
                time.sleep(rate_delay)

        # ── Validasi: pastikan setidaknya ada data ────────────────────
        if not all_raw:
            raise ValueError(
                f"Exchange tidak mengembalikan data apapun untuk "
                f"{symbol} [{timeframe}]. Periksa simbol dan koneksi jaringan."
            )

        # ── Bangun DataFrame dari semua batch ─────────────────────────
        df = pd.DataFrame(
            all_raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # Konversi timestamp milidetik → datetime UTC dan jadikan index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Vectorized cast ke float
        df = df.astype(float)

        # ── Pembersihan Data ──────────────────────────────────────────
        # 1. Hapus baris duplikat (bisa terjadi di boundary antar batch)
        rows_before_dedup = len(df)
        df = df[~df.index.duplicated(keep="last")]
        dupes_removed = rows_before_dedup - len(df)
        if dupes_removed > 0:
            logger.info(
                "[DataFetcher] %d baris duplikat dihapus.", dupes_removed
            )

        # 2. Urutkan kronologis: terlama → terbaru (ascending)
        df.sort_index(inplace=True)

        # 3. Potong ke jumlah tepat yang diminta (ambil N candle terakhir)
        if len(df) > limit:
            df = df.iloc[-limit:]

        logger.info(
            "[DataFetcher] Selesai: %d candle dalam %d batch | %s → %s",
            len(df),
            batch_num,
            df.index[0].strftime("%Y-%m-%d %H:%M UTC"),
            df.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
        )

        return df
