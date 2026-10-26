#!/usr/bin/env bash
# AWS EC2 one-time setup for the Pocket Option Signal Bot.
# Works from anywhere — clones the repo wherever you cloned it.
# Usage: sudo bash deploy/setup.sh
set -euo pipefail

# Project root = parent of the deploy/ folder containing this script
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> [0/5] Git state"
git -C "${APP_DIR}" rev-parse --short HEAD 2>/dev/null || echo "  (not a git checkout)"

echo "==> [1/5] Installing system packages"
apt-get update -y
apt-get install -y python3-venv python3-pip

echo "==> [2/5] Verifying bot token present in ${APP_DIR}/.env"
if [ ! -f "${APP_DIR}/.env" ]; then
  echo "ERROR: ${APP_DIR}/.env missing (it is git-ignored on purpose)."
  echo "Create it on the server:  nano ${APP_DIR}/.env"
  echo "Then paste:               BOT_TOKEN=your_botfather_token"
  echo "Or copy it from your PC:  scp -i KEY.pem .env ubuntu@SERVER_IP:${APP_DIR}/.env"
  exit 1
fi
if ! grep -q "^BOT_TOKEN=" "${APP_DIR}/.env"; then
  echo "ERROR: .env has no BOT_TOKEN line."
  exit 1
fi

echo "==> [3/5] Creating virtualenv + installing requirements"
python3 -m venv "${APP_DIR}/venv"
"${APP_DIR}/venv/bin/pip" install --upgrade pip
"${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

echo "==> [4/5] Installing systemd service"
sed "s|%APP_DIR%|${APP_DIR}|g" "${APP_DIR}/deploy/po-bot.service" |
  tee /etc/systemd/system/po-bot.service > /dev/null

echo "==> [5/5] Enabling + starting service"
systemctl daemon-reload
systemctl enable --now po-bot
sleep 3
systemctl --no-pager status po-bot || true

echo
echo "Deployed. Useful commands:"
echo "  systemctl status po-bot        # is it running"
echo "  journalctl -u po-bot -f        # live logs (startup connectivity check shows here)"
echo "  sudo systemctl restart po-bot  # after code changes"
