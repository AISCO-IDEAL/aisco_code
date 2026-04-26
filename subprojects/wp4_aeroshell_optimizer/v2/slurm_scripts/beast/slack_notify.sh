#!/bin/bash
# TS_ONFINISH hook + manual notifier for the AISCO-Beast shared GPU queue.
#
# Called automatically by tsp when a job finishes:
#     slack_notify.sh <job_id> <exit_code> <output_file> <command>
#
# Can also be invoked manually for tests / queued-position pings:
#     slack_notify.sh --queued <slack_user> <job_id> <position> <command>
#     slack_notify.sh --test   <slack_user> <message>

set -euo pipefail

# ---- config ---------------------------------------------------------------
TOKEN_FILE="${SLACK_TOKEN_FILE:-/home/aisco/.slack_token}"   # contains the bot token, mode 0600
MAPPING_FILE="${SLACK_MAPPING_FILE:-/etc/AISCO-Beast_mapping.conf}"
QUEUE_SOCKET="${TS_SOCKET:-/tmp/gpu_queue}"
TSP_BIN="/usr/bin/tsp"
LOG_FILE="${SLACK_NOTIFY_LOG:-/var/log/aisco/slack_notify.log}"
META_PREFIX="/tmp/aisco-tsp-meta"

mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || LOG_FILE="/tmp/slack_notify.log"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" >>"$LOG_FILE" 2>/dev/null || true; }

if [ ! -r "$TOKEN_FILE" ]; then
    log "ERROR: cannot read token file $TOKEN_FILE"
    echo "ERROR: cannot read token file $TOKEN_FILE" >&2
    exit 0   # never block tsp
fi
TOKEN="$(tr -d '[:space:]' <"$TOKEN_FILE")"

# ---- helpers --------------------------------------------------------------
lookup_slack_id() {
    local username="$1"
    awk -F: -v u="$username" '$1 == u { print $2; exit }' "$MAPPING_FILE"
}

send_slack_message() {
    local slack_id="$1" message="$2" payload resp
    payload=$(python3 - "$slack_id" "$message" <<'PY'
import json, sys
print(json.dumps({"channel": sys.argv[1], "text": sys.argv[2]}))
PY
)
    resp=$(curl -sS -X POST "https://slack.com/api/chat.postMessage" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "$payload" || true)
    if ! printf '%s' "$resp" | grep -q '"ok":true'; then
        log "Slack API error for $slack_id: $resp"
    else
        log "Slack DM sent to $slack_id"
    fi
}

job_info() {
    TS_SOCKET="$QUEUE_SOCKET" "$TSP_BIN" -i "$1" 2>/dev/null || true
}

job_value() {
    awk -F= -v key="$1" '$1 == key { print substr($0, index($0, "=") + 1); exit }'
}

command_value() {
    local key="$1" command_line="$2"
    printf '%s\n' "$command_line" | sed -n "s/.*[[:space:]]${key}=\([^[:space:]]*\).*/\1/p" | head -n1
}

resolve_recipient() {
    local info="$1" r
    r=$(printf '%s\n' "$info" | job_value "AISCO_NOTIFY_USERNAME")
    [ -n "$r" ] && { printf '%s\n' "$r"; return; }
    r=$(printf '%s\n' "$info" | job_value "AISCO_SUBMITTER")
    [ -n "$r" ] && { printf '%s\n' "$r"; return; }
    printf '\n'
}

meta_value() {
    local key="$1" job_id="$2" meta_file
    meta_file="${META_PREFIX}-${job_id}.env"
    [ -r "$meta_file" ] || return 0
    awk -F= -v k="$key" '$1 == k { print substr($0, index($0, "=") + 1); exit }' "$meta_file"
}

notify_user() {
    local user="$1" message="$2" slack_id
    slack_id=$(lookup_slack_id "$user")
    if [ -z "$slack_id" ]; then
        log "No Slack mapping for '$user'"
        echo "No Slack mapping found for recipient $user" >&2
        return 0
    fi
    send_slack_message "$slack_id" "$message"
}

# ---- manual modes ---------------------------------------------------------
case "${1:-}" in
    --test)
        RECIPIENT=${2:?usage: --test <user> <message>}
        MESSAGE=${3:-test from $(hostname)}
        notify_user "$RECIPIENT" "$MESSAGE"
        exit 0
        ;;
    --queued)
        RECIPIENT=${2:?missing recipient}
        JOB_ID=${3:?missing job id}
        POSITION=${4:?missing queue position}
        COMMAND=${5:-}
        MSG=$'*GPU JOB QUEUED* on AISCO-Beast'
        MSG+=$'\n*User:* '"$RECIPIENT"
        MSG+=$'\n*Job ID:* '"$JOB_ID"
        MSG+=$'\n*Queue position:* '"$POSITION"
        MSG+=$'\n*Command:* '"'$COMMAND'"
        notify_user "$RECIPIENT" "$MSG"
        exit 0
        ;;
esac

# ---- finish-hook mode (default) -------------------------------------------
JOB_ID=${1:?missing job id}
EXIT_CODE=${2:?missing exit code}
OUTPUT_FILE=${3:-}
COMMAND=${4:-}

INFO=$(job_info "$JOB_ID")
RECIPIENT=$(resolve_recipient "$INFO")
SUBMITTER=$(printf '%s\n' "$INFO" | job_value "AISCO_SUBMITTER")

# Fallback for cases where tsp metadata is not available at hook time.
if [ -z "$RECIPIENT" ] || [ "$RECIPIENT" = "${USER:-unknown}" ]; then
    RECIPIENT=$(command_value "AISCO_NOTIFY_USERNAME" "$COMMAND")
fi
[ -z "$RECIPIENT" ] && RECIPIENT=$(meta_value "AISCO_NOTIFY_USERNAME" "$JOB_ID")
[ -z "$RECIPIENT" ] && RECIPIENT="${USER:-unknown}"

if [ -z "$SUBMITTER" ]; then
    SUBMITTER=$(command_value "AISCO_SUBMITTER" "$COMMAND")
fi
[ -z "$SUBMITTER" ] && SUBMITTER=$(meta_value "AISCO_SUBMITTER" "$JOB_ID")
[ -z "$SUBMITTER" ] && SUBMITTER="$RECIPIENT"

if [ "$EXIT_CODE" -eq 0 ]; then
    STATUS="Success"
else
    STATUS="Failed (exit $EXIT_CODE)"
fi

MSG=$'*GPU JOB FINISHED* on AISCO-Beast'
MSG+=$'\n*User:* '"$RECIPIENT"
MSG+=$'\n*Submitted by:* '"$SUBMITTER"
MSG+=$'\n*Job ID:* '"$JOB_ID"
MSG+=$'\n*Status:* '"$STATUS"
MSG+=$'\n*Command:* '"'$COMMAND'"
MSG+=$'\n*Output:* '"$OUTPUT_FILE"

notify_user "$RECIPIENT" "$MSG"
rm -f "${META_PREFIX}-${JOB_ID}.env" 2>/dev/null || true
