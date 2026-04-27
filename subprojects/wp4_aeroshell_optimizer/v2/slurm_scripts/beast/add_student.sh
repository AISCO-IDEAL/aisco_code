#!/bin/bash
# Provision a new BEAST user + Slack mapping. Run as a sudoer.
#
# Usage: add_student.sh <username> <slack_id> <temp_password>

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <username> <slack_id> <temp_password>" >&2
    exit 1
fi

USERNAME=$1
SLACKID=$2
PASSWORD=$3
MAPPING="/etc/AISCO-Beast_mapping.conf"

if id "$USERNAME" >/dev/null 2>&1; then
    echo "User $USERNAME already exists; refreshing password and Slack mapping."
else
    sudo useradd -m -s /bin/bash "$USERNAME"
fi

echo "$USERNAME:$PASSWORD" | sudo chpasswd
sudo chage -d 0 "$USERNAME"

# Update / append Slack mapping atomically
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
sudo awk -F: -v u="$USERNAME" -v s="$SLACKID" '
    BEGIN { done = 0 }
    $1 == u { print u ":" s; done = 1; next }
    { print }
    END { if (!done) print u ":" s }
' "$MAPPING" > "$tmp"
sudo install -m 644 "$tmp" "$MAPPING"

# Make sure new user gets the right env on login
PROFILE_SNIPPET=$(cat <<'EOF'

# --- AISCO-Beast shared GPU queue (auto-added) ---
export TS_SOCKET=/tmp/gpu_queue
alias bs='/home/aisco/beast-submit'
EOF
)
USER_HOME=$(getent passwd "$USERNAME" | cut -d: -f6)
if [ -n "$USER_HOME" ] && [ -d "$USER_HOME" ]; then
    if ! sudo grep -q 'AISCO-Beast shared GPU queue' "$USER_HOME/.bashrc" 2>/dev/null; then
        printf '%s\n' "$PROFILE_SNIPPET" | sudo tee -a "$USER_HOME/.bashrc" >/dev/null
        sudo chown "$USERNAME:$USERNAME" "$USER_HOME/.bashrc"
    fi
fi

echo "------------------------------------------"
echo "Success! User $USERNAME is ready."
echo "  Slack ID : $SLACKID"
echo "  Submit   : /home/aisco/beast-submit $USERNAME <command...>"  # or just: beast-submit if symlinked"
echo "  Inspect  : tsp"
echo "  Password forced to change on first login."
