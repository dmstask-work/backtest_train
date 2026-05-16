#!/bin/bash
# Script instalasi otomatis VPS Ubuntu (GCP / AWS / DO)

echo "[1/4] Mengupdate Ubuntu dan menginstall pustaka dasar..."
sudo apt update && sudo apt install -y python3-venv python3-pip git curl

echo "[2/4] Menyiapkan Virtual Environment (bot-env)..."
python3 -m venv bot-env
source bot-env/bin/activate

echo "[3/4] Menginstall depedensi Python (requirements.txt)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Memastikan folder logs/ dan data/ ada..."
mkdir -p logs data
cp .env.example .env

echo ""
echo "✅ INSTALASI VPS SELESAI!"
echo "============================================================"
echo "Langkah selanjutnya yang perlu Anda ketik di terminal VPS:"
echo "1. Edit file kunci API  : nano .env"
echo "2. Tes dry-run manual  : source .env && bot-env/bin/python live_bot.py --dry-run"
echo "3. Pasang automasi     : crontab -e"
echo "============================================================"