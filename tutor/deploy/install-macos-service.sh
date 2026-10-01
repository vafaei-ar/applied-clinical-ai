#!/usr/bin/env bash
# Run the tutor bot in the background on macOS with launchd: it starts at login and restarts if
# it crashes. Usage: ./deploy/install-macos-service.sh [uninstall]
set -euo pipefail

LABEL="org.appliedclinicalai.tutor"
TUTOR_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
BIN="$TUTOR_DIR/.venv/bin/clinical-tutor"

if [[ "${1:-}" == "uninstall" ]]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL."
  exit 0
fi

[[ -x "$BIN" ]] || { echo "Install first: cd $TUTOR_DIR && uv venv && uv pip install -e ."; exit 1; }
[[ -f "$TUTOR_DIR/.env" ]] || { echo "Create $TUTOR_DIR/.env from .env.example first."; exit 1; }
mkdir -p "$TUTOR_DIR/logs" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <!-- caffeinate -i keeps the Mac from idle-sleeping while the bot runs (display may sleep). -->
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>$BIN</string><string>run</string>
  </array>
  <key>WorkingDirectory</key><string>$TUTOR_DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$TUTOR_DIR/logs/tutor.log</string>
  <key>StandardErrorPath</key><string>$TUTOR_DIR/logs/tutor.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed and started $LABEL. Logs: $TUTOR_DIR/logs/tutor.log"
echo "Stop:  launchctl bootout gui/$(id -u)/$LABEL    Remove: $0 uninstall"
