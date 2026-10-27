# Deploying the Pocket Option Signal Bot on AWS (24/7)

Two supported ways to get the code to the server:

- **Route A — GitHub (recommended, no local install):** upload via github.com in
  your browser, then `git clone` on the server. Updates = re-upload + `git pull`.
- **Route B — direct upload:** zip + `scp` (no GitHub at all).

Both end in the same place: systemd service running the bot 24/7.

> ⚠️ **Rule #1: never upload your `.env` to GitHub.** It contains your bot token.
> The repo files here already exclude it — keep it that way. The token lives only
> on your PC and the server.

---

## Route A — GitHub (no Git install needed)

### 1. Create the repo

1. Go to <https://github.com/new>
2. Repo name: e.g. `pocketoption-signal-bot`
3. Visibility: **Private** ✅ (keep your strategy private)
4. Create (empty repo — don't add README on the website).

### 2. Upload the project from your browser

1. Open the repo → **"uploading an existing file"** link (or Add file → Upload files).
2. Drag in the **contents** of `C:\Users\sigma\Documents\Default Project\pocketoption-signal-bot`
   (main.py, engine/, deploy/, requirements.txt, README.md, .gitignore, .env.example — **NOT .env**).
3. Commit changes (button at the bottom).

Update flow later: same page → re-upload changed files → commit. GitHub replaces
same-named files.

### 3. Let the server clone it (needs a Personal Access Token)

A **fine-grained PAT** authenticates the private clone:

1. GitHub → Settings → Developer settings → **Personal access tokens (fine-grained)**
2. Generate new token:
   - Repository access: **Only select repositories** → pick `pocketoption-signal-bot`
   - Permissions: Repository → **Contents: Read-only** (that's all the server needs)
   - Expiration: as long as you like
3. Copy the token (`github_pat_...`) — shown once.

### 4. Server: clone + deploy

SSH in:

```bash
sudo apt-get install -y git unzip
git clone https://github.com/<YOUR_USERNAME>/pocketoption-signal-bot.git /opt/pocketoption-signal-bot
# When it asks for credentials:
#   Username: your GitHub username
#   Password: paste the PAT (not your GitHub password!)
```

Create the token file **on the server** (it was never in git):

```bash
cd /opt/pocketoption-signal-bot
nano .env        # paste line:  BOT_TOKEN=your_botfather_token   → save with Ctrl+O, Enter, Ctrl+X
```

Deploy:

```bash
sudo bash deploy/setup.sh
journalctl -u po-bot -f      # expect: LIVE FEED REACHABLE ✅
```

Or pre-load `.env` from your PC in one step (PowerShell, on your machine):

```powershell
scp -i aws-ec2.pem "C:\Users\sigma\Documents\Default Project\pocketoption-signal-bot\.env" ubuntu@SERVER_IP:/opt/pocketoption-signal-bot/.env
```

### 5. Update later (edit → upload → pull → restart)

```bash
cd /opt/pocketoption-signal-bot
sudo git pull
sudo systemctl restart po-bot
```

(If `git pull` refuses because of local changes: `sudo git checkout -- .` first —
your local copy should always stay a pure mirror of GitHub.)

If the PAT expires or you rotate it: `sudo git remote set-url origin https://<USER>:<NEW_PAT>@github.com/<USER>/pocketoption-signal-bot.git`

### 6. Live OTC feed (ssid) on the server

The broker feed needs the `BinaryOptionsToolsV2` package (installed by
`requirements.txt`) and a valid session string:

```bash
cd /opt/pocketoption-signal-bot
sudo venv/bin/pip install -r requirements.txt
sudo venv/bin/python -m engine.po_feed --save '42["auth",{...}]'
sudo venv/bin/python -m engine.po_feed --show      # assets + EURUSD-OTC payout
sudo systemctl restart po-bot
journalctl -u po-bot -f                            # expect: READY — LIVE OTC CANDLES ENABLED
```

`pocket_ssid.json` is git-ignored (never upload it). `PO_SSID` in `.env` overrides
the file if both exist. Verify freshness with
`sudo venv/bin/python -m engine.po_feed --candles EUR/USD` — it prints the last
bar's age in seconds (a healthy feed stays under ~90 s).

Run the full live check after deploying (it requests every OTC pair, ~4 min):

```bash
sudo venv/bin/python test_live_check.py
```

If the live check reports `payout n/a` on most pairs, dump the raw payout shape
(no secrets are printed) and keep the output:

```bash
sudo venv/bin/python tools/payout_probe.py EURUSD_otc
```

It shows the bulk `payout()` type/size, the per-symbol lookup, and the payout
field in the active-asset list, plus the exact library/pandas versions.

Pocket Option rotates its OTC line-up. A pair that is not currently tradeable is
**refused** by the bot with an explanation instead of being analysed — that is
expected, not a bug. Re-run the live check later to see the current list.

---

## Route B — scp zip (no GitHub)

```powershell
cd "C:\Users\sigma\Documents\Default Project"
Compress-Archive -Path pocketoption-signal-bot -DestinationPath po-bot.zip
scp -i aws-ec2.pem po-bot.zip ubuntu@SERVER_IP:~/
```

Server:

```bash
sudo unzip -o ~/po-bot.zip -d /opt/ && sudo mv /opt/pocketoption-signal-bot* /opt/pocketoption-signal-bot 2>/dev/null || true
sudo bash /opt/pocketoption-signal-bot/deploy/setup.sh
```

Updates: re-zip, re-scp, re-unzip, `sudo systemctl restart po-bot`.

---

## 1. Create the AWS free-tier EC2 instance

In the AWS Console → EC2:

1. **Launch instance**
   - AMI: **Ubuntu Server 24.04 LTS** (Free tier eligible)
   - Type: **t3.micro** (or t2.micro — 24/7 free tier)
   - Key pair: create one → download the `.pem` (e.g. `aws-ec2.pem`)
   - Security group: allow **SSH (22) from "My IP" only**. Nothing else inbound.
   - Storage: 8 GB gp3
2. Launch, wait for **Running**, copy the **Public IPv4**.

PowerShell key permissions (one time, from your PC):

```powershell
icacls aws-ec2.pem /inheritance:r /grant:r "$env:USERNAME:R"
ssh -i aws-ec2.pem ubuntu@SERVER_IP
```

Bot architecture note: the bot only makes **outbound** HTTPS calls (Telegram polling +
Yahoo Finance). No inbound ports beyond your SSH access.

## Day-2 operations

| Task | Command (on the server) |
|---|---|
| Is it running? | `systemctl status po-bot` |
| Live logs | `journalctl -u po-bot -f` |
| Restart after code change | `sudo systemctl restart po-bot` |
| Stop / start | `sudo systemctl stop po-bot` / `start` |
| Feed health in chat | send `/status` to the bot in Telegram |

## Cost / limits

- **t3.micro (or t2.micro) 24/7** is covered by the AWS Free Tier for the first
  12 months of the account; after that it's roughly ~$7.5/mo.
- Set a **billing alarm** (Billing → Budgets → $5) as a safety net.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Clone asks for password, token rejected | PAT missing Contents:Read-only, or wrong repo selected |
| `journalctl` shows `NO INTERNET` instantly | `ping -c2 api.telegram.org` — check instance outbound access |
| Bot never online in Telegram | `grep BOT_TOKEN /opt/pocketoption-signal-bot/.env`; regenerate in BotFather if needed |
| Fresh start | `sudo systemctl stop po-bot && sudo rm -rf venv && sudo bash deploy/setup.sh` |
