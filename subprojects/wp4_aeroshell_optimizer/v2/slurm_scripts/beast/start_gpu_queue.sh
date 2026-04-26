#!/bin/bash
# Starts the shared Task Spooler daemon on AISCO-Beast.
#
# Run this AS USER 'aisco' once after each reboot (or via systemd user unit /
# crontab @reboot). All other users connect to the same daemon by setting
# TS_SOCKET=/tmp/gpu_queue.
#
# Single-slot queue => exactly one heavy job runs at a time.
# TS_ONFINISH points at slack_notify.sh so every finished job DMs its owner.

set -euo pipefail

QUEUE_SOCKET="/tmp/gpu_queue"
NOTIFIER="/home/aisco/slack_notify.sh"
REAL_TSP="/usr/bin/tsp"

if [ "$(id -un)" != "aisco" ]; then
    echo "ERROR: start_gpu_queue.sh must be run as user 'aisco'." >&2
    exit 1
fi

# Clean up a stale socket file (regular file or socket owned by us)
if [ -e "$QUEUE_SOCKET" ] && [ ! -S "$QUEUE_SOCKET" ]; then
    rm -f "$QUEUE_SOCKET"
fi
if [ -S "$QUEUE_SOCKET" ] && ! TS_SOCKET="$QUEUE_SOCKET" "$REAL_TSP" >/dev/null 2>&1; then
    # Daemon dead but socket file lingering
    rm -f "$QUEUE_SOCKET"
fi

# Already up?
if TS_SOCKET="$QUEUE_SOCKET" "$REAL_TSP" >/dev/null 2>&1; then
    echo "Queue already running on $QUEUE_SOCKET"
else
    TS_SOCKET="$QUEUE_SOCKET" \
    TS_ONFINISH="$NOTIFIER" \
    TS_MAXFINISHED=200 \
    "$REAL_TSP" -S 1
    echo "Queue started on $QUEUE_SOCKET (1 concurrent slot, on-finish=$NOTIFIER)"
fi

# World-writable socket so other users can submit / inspect
chmod 0666 "$QUEUE_SOCKET"

echo
echo "Users on this machine should add to ~/.bashrc:"
echo "    export TS_SOCKET=$QUEUE_SOCKET"
echo
echo "Submission helper: /home/aisco/beast-submit <slack-user> <command...>"
echo "Inspect queue:     tsp           tsp -i <ID>     tail -f \$(tsp -o <ID>)"
