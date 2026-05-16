"""
Modul Pengambil Data OHLCV dengan Local Cache (v3).

Mengambil data candlestick historis dari exchange menggunakan library ccxt,
dengan mekanisme Local Disk Cache + Delta Fetching untuk mencegah rate-limit
selama siklus optimasi otomatis.

Strategi Caching:
  1. Format    : Parquet (pyarrow/fastparquet) jika tersedia, fallback ke CSV.
  2. Lokasi    : <project_root>/data/{SYMBOL}_{timeframe}.parquet|.csv
                 (karakter '/' pada simbol diganti '_', contoh: SOL_USDT_4h)
  3. Cold Start: Tidak ada cache → ambil penuh dari exchange, simpan ke disk.
  4. Delta Fetch: Cache ada → muat file, baca timestamp terakhir, ambil HANYA
                 candle baru dari exchange, gabung + dedup + simpan ulang.
  5. Output    : DataFrame selalu dipotong ke `limit` baris terbaru sehingga
                 pemanggil mendapat tepat apa yang diminta.

v3 vs v2:
  • Tambah _cache_path(), _build_df(), _load_cache(), _save_cache()
  • Refactor pagination menjadi _fetch_raw_since() (metode internal)
  • fetch_ohlcv() menjadi orkestrator: cache → delta/full → trim → return
  • Index tetap tz-naive UTC DatetimeIndex (kompatibel dengan BacktestEngine)
  • Tidak ada perubahan pada config.py atau modul lainnya
"""

import importlib.util
import logging
import time
from pathlib import Path

import ccxt
import pandas as pd

from config import EXCHANGE_CONFIG, EXCHANGE_MODE

logger = logging.getLogger(__name__)

# Jumlah candle maksimum per satu request (batas Binance)
_BATCH_SIZE: int = 1000

# Konfigurasi retry untuk menangani gangguan jaringan transien
_MAX_RETRIES: int   = 3    # Maksimum percobaan ulang per batch
_RETRY_DELAY: float = 5.0  # Jeda antar percobaan ulang (detik)

# ── Cache Setup ───────────────────────────────────────────────────────────────
# Direktori cache relatif terhadap lokasi file ini (bukan cwd) agar aman
# dijalankan dari crontab atau direktori manapun.
_CACHE_DIR: Path = Path(__file__).parent / "data"

# Gunakan Parquet jika pyarrow atau fastparquet tersedia; jika tidak, CSV.
_USE_PARQUET: bool = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Fungsi Cache (module-level, tidak butuh instance exchange)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(symbol: str, timeframe: str) -> Path:
    """
    Mengembalikan path file cache untuk pasangan symbol + timeframe.

    '/' dalam nama simbol diganti '_' agar aman sebagai nama file
    (contoh: SOL/USDT → SOL_USDT → data/SOL_USDT_4h.parquet).

    Args:
        symbol:    Simbol pasangan trading, contoh: 'SOL/USDT'.
        timeframe: Timeframe candle, contoh: '4h'.

    Returns:
        Path objek ke file cache (belum tentu ada di disk).
    """
    safe = symbol.replace("/", "_")
    ext  = ".parquet" if _USE_PARQUET else ".csv"
    return _CACHE_DIR / f"{safe}_{timeframe}{ext}"


def _build_df(raw: list) -> pd.DataFrame:
    """
    Mengkonversi list OHLCV mentah dari ccxt menjadi DataFrame berindex waktu.

    Timestamp dikonversi dari milidetik epoch ke datetime tz-naive UTC.
    Tidak menggunakan utc=True agar index tetap tz-naive (DatetimeTZDtype
    dihindari karena tidak konsisten dengan pembacaan ulang dari CSV).

    Args:
        raw: List of lists dari ccxt.fetch_ohlcv() —
             setiap item: [timestamp_ms, open, high, low, close, volume].

    Returns:
        DataFrame dengan tz-naive UTC DatetimeIndex bernama 'timestamp'
        dan kolom float64: open, high, low, close, volume.
    """
    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    # unit='ms' tanpa tz= menghasilkan tz-naive UTC (epoch ms → UTC datetime)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df


def _load_cache(path: Path) -> "pd.DataFrame | None":
    """
    Memuat DataFrame dari file cache (Parquet atau CSV).

    Menormalkan index menjadi tz-naive DatetimeIndex setelah memuat,
    sehingga data dari versi lama (tz-aware) tetap kompatibel.

    Args:
        path: Path ke file cache yang akan dimuat.

    Returns:
        DataFrame yang sudah bersih, atau None jika memuat gagal.
    """
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            # index_col='timestamp' agar index bernama konsisten
            df = pd.read_csv(path, index_col="timestamp", parse_dates=True)

        if df is None or df.empty:
            return None

        # ── Normalise index → tz-naive UTC DatetimeIndex ──────────────
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Jika cache lama menyimpan tz-aware, konversi ke tz-naive UTC
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df.index.name = "timestamp"
        df.sort_index(inplace=True)

        logger.info(
            "[DataFetcher] Cache dimuat: %s → %d baris (%s hingga %s)",
            path.name,
            len(df),
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )
        return df

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[DataFetcher] Gagal memuat cache '%s': %s — akan fetch ulang.",
            path.name,
            exc,
        )
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """
    Menyimpan DataFrame ke file cache (Parquet atau CSV).

    Direktori dibuat otomatis jika belum ada. Error saat simpan hanya
    dicatat sebagai warning — tidak menghentikan eksekusi program.

    Args:
        df:   DataFrame yang akan disimpan (index = tz-naive DatetimeIndex).
        path: Tujuan file cache.
    """
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            df.to_parquet(path)
        else:
            df.to_csv(path)
        logger.info(
            "[DataFetcher] Cache disimpan → %s  (%d baris)", path.name, len(df)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[DataFetcher] Gagal menyimpan cache '%s': %s", path.name, exc
        )


# ─────────────────────────────────────────────────────────────────────────────
# Kelas Utama
# ─────────────────────────────────────────────────────────────────────────────

class DataFetcher:
    """
    Kelas untuk mengambil data OHLCV historis dari exchange dengan cache lokal.

    Alur kerja:
      1. Periksa apakah file cache ada di data/.
      2. Jika ada → muat cache, ambil HANYA candle delta (terbaru).
      3. Jika tidak ada → ambil penuh dari exchange, simpan ke cache.
      4. Kembalikan N baris terbaru sesuai `limit`.

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
                # Tingkatkan timeout agar tidak langsung gagal pada jaringan lambat
                "timeout": 30000,  # 30 detik
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
    # Pagination Primitif (Internal)
    # ------------------------------------------------------------------
    def _fetch_raw_since(
        self,
        symbol:    str,
        timeframe: str,
        since_ms:  int,
        limit:     int,
    ) -> list:
        """
        Mengambil candle dari exchange mulai dari since_ms, hingga `limit` candle.

        Ini adalah metode primitif yang hanya menangani komunikasi dengan
        exchange (pagination + rate-limit). Tidak ada logika cache di sini.
        Berhenti lebih awal jika exchange mengembalikan batch lebih kecil
        dari yang diminta (tidak ada lagi data historis).

        Args:
            symbol:    Simbol pasangan trading, contoh: 'SOL/USDT'.
            timeframe: Timeframe candle, contoh: '4h'.
            since_ms:  Timestamp awal dalam milidetik (epoch UTC).
            limit:     Jumlah maksimum candle yang ingin diambil.

        Returns:
            List of lists OHLCV mentah dari ccxt (bisa lebih sedikit dari limit).

        Raises:
            ccxt.NetworkError:  Saat terjadi kesalahan jaringan.
            ccxt.ExchangeError: Saat terjadi kesalahan dari sisi exchange.
        """
        tf_ms:       int   = self.exchange.parse_timeframe(timeframe) * 1000
        rate_delay:  float = max(self.exchange.rateLimit / 1000, 0.5)

        all_raw:       list = []
        batch_num:     int  = 0
        total_batches: int  = max(-(-limit // _BATCH_SIZE), 1)  # ceiling division
        current_since: int  = since_ms

        while len(all_raw) < limit:
            remaining = limit - len(all_raw)
            to_fetch  = min(_BATCH_SIZE, remaining)
            batch_num += 1

            logger.info(
                "[DataFetcher] Batch %d/%d | Mengambil %d candle sejak %s ...",
                batch_num,
                total_batches,
                to_fetch,
                pd.Timestamp(current_since, unit="ms").strftime("%Y-%m-%d %H:%M"),
            )

            # ── Fetch dengan retry + exponential-ish backoff ──────────
            batch: list = []
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    batch = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=current_since,
                        limit=to_fetch,
                    )
                    break  # sukses — keluar dari loop retry
                except (
                    ccxt.RequestTimeout,
                    ccxt.NetworkError,
                    ccxt.ExchangeError,
                ) as exc:
                    if attempt < _MAX_RETRIES:
                        logger.warning(
                            "[DataFetcher] Error jaringan pada batch %d "
                            "(percobaan %d/%d): %s — retry dalam %.0f detik ...",
                            batch_num, attempt, _MAX_RETRIES,
                            exc, _RETRY_DELAY,
                        )
                        time.sleep(_RETRY_DELAY)
                    else:
                        logger.error(
                            "[DataFetcher] Batch %d GAGAL setelah %d percobaan: %s",
                            batch_num, _MAX_RETRIES, exc,
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
            current_since = batch[-1][0] + tf_ms

            # Exchange mengembalikan lebih sedikit dari yang diminta →
            # tidak ada lagi data historis di periode ini
            if len(batch) < to_fetch:
                logger.info(
                    "[DataFetcher] Batch %d: %d < %d candle. "
                    "Batas data historis tercapai.",
                    batch_num, len(batch), to_fetch,
                )
                break

            # Rate-limit delay — WAJIB jika masih ada batch berikutnya
            if len(all_raw) < limit:
                logger.debug(
                    "[DataFetcher] Rate-limit delay %.2f detik ...", rate_delay
                )
                time.sleep(rate_delay)

        return all_raw

    # ------------------------------------------------------------------
    # Fungsi Publik Utama — Orkestrator Cache + Fetch
    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int,
    ) -> pd.DataFrame:
        """
        Mengambil data OHLCV historis dengan cache lokal + delta fetching.

        Alur kerja:
          [Cache HIT]
            1. Muat DataFrame dari data/<symbol>_<tf>.parquet|.csv
            2. Baca timestamp candle terakhir di cache.
            3. Estimasi jumlah candle baru (delta) sejak timestamp tersebut.
            4. Ambil HANYA candle delta dari exchange (hemat kuota API).
            5. Concat + dedup (keep='last') + sort kronologis.
            6. Simpan ulang (overwrite) file cache.

          [Cache MISS]
            1. Ambil penuh `limit` candle menggunakan pagination.
            2. Simpan ke file cache baru.

          [Output]
            • Selalu dipotong ke `limit` baris terbaru.
            • Index: tz-naive UTC DatetimeIndex bernama 'timestamp'.
            • Kolom: open, high, low, close, volume (semua float64).

        Args:
            symbol:    Simbol pasangan trading, contoh: 'SOL/USDT'.
            timeframe: Timeframe candle, contoh: '4h'.
            limit:     Total candle yang dikembalikan (N baris terbaru).

        Returns:
            DataFrame dengan index timestamp (UTC, tz-naive) dan kolom OHLCV.
            Jumlah baris = min(limit, data tersedia).

        Raises:
            ccxt.NetworkError:  Saat terjadi kesalahan jaringan.
            ccxt.ExchangeError: Saat terjadi kesalahan dari sisi exchange.
            ValueError:         Jika tidak ada cache dan exchange kosong.
        """
        cache_path = _cache_path(symbol, timeframe)
        tf_ms: int = self.exchange.parse_timeframe(timeframe) * 1000

        logger.info(
            "[DataFetcher] fetch_ohlcv(%s, %s, limit=%d) | "
            "Format cache: %s | File: %s",
            symbol,
            timeframe,
            limit,
            "parquet" if _USE_PARQUET else "csv",
            cache_path.name,
        )

        # ── Coba Muat Cache ───────────────────────────────────────────
        df_cached: "pd.DataFrame | None" = None
        if cache_path.exists():
            df_cached = _load_cache(cache_path)

        # ── JALUR A: Cache HIT — Delta Fetch ─────────────────────────
        if df_cached is not None and not df_cached.empty:
            last_ts_ms: int = int(df_cached.index[-1].timestamp() * 1000)
            now_ms:     int = self.exchange.milliseconds()

            # Estimasi candle dalam delta; +2 = candle berjalan + safety margin.
            # Mulai dari last_ts_ms (inklusif) agar candle terakhir yang
            # mungkin masih terbentuk diperbarui dengan data final.
            delta_est: int = max(int((now_ms - last_ts_ms) / tf_ms) + 2, 1)

            logger.info(
                "[DataFetcher] Cache HIT: %d baris dimuat. "
                "Mengambil delta ~%d candle baru ...",
                len(df_cached),
                delta_est,
            )

            raw_delta = self._fetch_raw_since(
                symbol, timeframe, last_ts_ms, delta_est
            )

            if raw_delta:
                df_delta  = _build_df(raw_delta)
                new_count = (~df_delta.index.isin(df_cached.index)).sum()

                df = pd.concat([df_cached, df_delta])
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)

                _save_cache(df, cache_path)
                logger.info(
                    "[DataFetcher] Delta selesai: +%d candle baru | "
                    "Total cache: %d baris.",
                    new_count,
                    len(df),
                )
            else:
                df = df_cached
                logger.info(
                    "[DataFetcher] Tidak ada candle baru dari exchange. "
                    "Menggunakan cache penuh (%d baris).",
                    len(df_cached),
                )

        # ── JALUR B: Cache MISS — Full Fetch ─────────────────────────
        else:
            logger.info(
                "[DataFetcher] Cache MISS. "
                "Mengambil %d candle penuh dari exchange ...",
                limit,
            )

            now_ms:   int = self.exchange.milliseconds()
            since_ms: int = now_ms - (limit * tf_ms)

            raw = self._fetch_raw_since(symbol, timeframe, since_ms, limit)

            if not raw:
                raise ValueError(
                    f"Exchange tidak mengembalikan data apapun untuk "
                    f"{symbol} [{timeframe}]. "
                    f"Periksa simbol dan koneksi jaringan."
                )

            df = _build_df(raw)
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

            _save_cache(df, cache_path)

        # ── Potong ke N Baris Terbaru yang Diminta ────────────────────
        if len(df) > limit:
            df = df.iloc[-limit:]

        logger.info(
            "[DataFetcher] Selesai: %d candle | %s → %s",
            len(df),
            df.index[0].strftime("%Y-%m-%d %H:%M UTC"),
            df.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
        )

        return df
