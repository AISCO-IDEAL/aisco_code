# AISCO-Beast Shared GPU Queue — minimal setup

This folder contains the **only** files needed to run a one-job-at-a-time GPU
queue on `AISCO-Beast` with Slack DM notifications and per-user log paths.

## Files in this folder

| File | Where it must live on BEAST | Owner | Purpose |
|---|---|---|---|
| `start_gpu_queue.sh` | `/home/aisco/start_gpu_queue.sh` | `aisco` | Starts the shared `tsp` daemon at `/tmp/gpu_queue` with `TS_ONFINISH=slack_notify.sh`. Run once after reboot (cron `@reboot` recommended). |
| `slack_notify.sh` | `/home/aisco/slack_notify.sh` | `aisco` | Called by `tsp` on each job finish; DMs the user mapped in `/etc/AISCO-Beast_mapping.conf`. Bot token lives in `/home/aisco/.slack_token` (mode 0600), **not** hard-coded. |
| `beast-submit` | `/home/aisco/beast-submit` | `aisco`, mode 0755 | The single submission entry point. Tags every job with `AISCO_NOTIFY_USERNAME` so the on-finish hook can find the recipient. |
| `add_student.sh` | anywhere a sudoer can run it | sudoer | Creates user, sets temp password (force-change on first login), updates Slack mapping, seeds `~/.bashrc`. |

## Files from the previous setup that are NO LONGER NEEDED

- `backfill_queue_access.sh`
- `refresh_queue_access.sh`
- `submit_remote_queue_job.sh`
- `submit_repo_branch_job.sh`
- `test_remote_queue.sh`
- `test_shared_queue.sh`
- `/var/tmp/aisco-queue/lib/shared_queue_common.sh` (and the `queue-remote-prepare`, `queue-remote-submit` helpers)

They implemented a multi-machine, branch-aware workflow that you don't need
for the "everyone runs `beast_testing.sh` on the one box" use-case. Delete or
archive them.

## One-time deployment on BEAST (as `aisco`)

In the current AISCO-Beast setup, assume:

- runnable repo: `/home/aisco/Documents/AISCO_code`
- Python virtualenv: `/home/aisco/Documents/AISCO_code/aisco_env`

```bash
# 0. Put the runnable repo on Beast
mkdir -p /home/aisco
# copy or clone the repo so beast_testing.sh exists at:
# /home/aisco/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast_testing.sh

# 1. Create the Slack bot token file used by slack_notify.sh
#    Get the bot token from your Slack app settings:
#    Slack API -> Your Apps -> OAuth & Permissions -> Bot User OAuth Token
#    The token starts with xoxb-
echo -n 'xoxb-your-rotated-bot-token' > /home/aisco/.slack_token
chmod 600 /home/aisco/.slack_token

# 2. Install the four scripts
install -m 0755 /home/aisco/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast/slack_notify.sh /home/aisco/slack_notify.sh
install -m 0755 /home/aisco/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast/start_gpu_queue.sh /home/aisco/start_gpu_queue.sh
mkdir -p /home/aisco/.local/bin
install -m 0755 /home/aisco/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast/beast-submit /home/aisco/beast-submit
sudo install -m 0755 /home/aisco/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast/add_student.sh /usr/local/sbin/add_student.sh

# 3. Make beast-submit reachable for everyone
sudo ln -sf /home/aisco/beast-submit /usr/local/bin/beast-submit

# 4. Logging dir for slack_notify
sudo mkdir -p /var/log/aisco
sudo chown aisco:aisco /var/log/aisco

# 5. Boot the queue, and persist via cron
/home/aisco/start_gpu_queue.sh
( crontab -l 2>/dev/null; echo "@reboot /home/aisco/start_gpu_queue.sh >>/var/log/aisco/queue.log 2>&1" ) | crontab -
```

If the queue was already running before `TS_ONFINISH` was configured, restart it explicitly:

```bash
/home/aisco/start_gpu_queue.sh --restart
```

## Per-user shell setup (auto-applied by `add_student.sh`)

Each user's `~/.bashrc` should contain:

```bash
export TS_SOCKET=/tmp/gpu_queue
alias bs='/usr/local/bin/beast-submit'
```

## Submitting a job

```bash
# use the Beast-local virtualenv for this launcher
export AISCO_VENV_DIR=/home/aisco/Documents/AISCO_code/aisco_env

# basic
beast-submit eschaerer bash /home/aisco/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast_testing.sh

# with a label and a custom log dir
export AISCO_VENV_DIR=/home/aisco/Documents/AISCO_code/aisco_env
BEAST_LOG_DIR=$HOME/runs/wp4 \
beast-submit -L wp4_test eschaerer bash /home/aisco/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast_testing.sh
```

`beast_testing.sh` resolves the Slack recipient in this order:
1. `AISCO_NOTIFY_USERNAME` (set automatically by `beast-submit`)
2. positional `$1`
3. `$USER`

So both `beast-submit eschaerer bash beast_testing.sh` and
`beast-submit eschaerer bash beast_testing.sh eschaerer` are correct; the env
var wins.

## Inspecting the queue

```bash
tsp                # list (works while a job is running)
tsp -i <ID>        # info, including AISCO_NOTIFY_USERNAME / AISCO_SUBMITTER
tail -f $(tsp -o <ID>)  # live tail without blocking the queue
tsp -t <ID>        # tail a finished job
tsp -r <ID>        # remove / kill
```

## Smoke tests

```bash
# 1. Slack works at all
/home/aisco/slack_notify.sh --test eschaerer "manual test"

# 2. Queue is alive and slack-on-finish fires
beast-submit eschaerer /bin/sh -c 'echo hi; sleep 5'
tsp                               # see it queued/running
# wait ~6s → DM "GPU JOB FINISHED" should arrive

# If the job runs but no DM arrives, restart the daemon with the notifier attached
/home/aisco/start_gpu_queue.sh --restart
beast-submit eschaerer /bin/sh -c 'echo hi; sleep 5'

# 3. Verify the launcher can see the chosen virtualenv
AISCO_VENV_DIR=/home/aisco/Documents/AISCO_code/aisco_env \
bash /home/aisco/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast_testing.sh eschaerer
```

## Why each of your three goals is now met

1. **Slack notifications work** — the daemon is *always* started by
   `start_gpu_queue.sh` with `TS_ONFINISH=/home/aisco/slack_notify.sh`, the
   token comes from a file readable only by `aisco`, and `slack_notify.sh`
   logs Slack API errors to `/var/log/aisco/slack_notify.log` instead of
   silently swallowing them.
2. **Each person submits their own `beast_testing.sh` and gets a queue slot** —
   `beast-submit` injects `AISCO_NOTIFY_USERNAME=<slack-user>` and
   `AISCO_SUBMITTER=$(id -un)` as env vars on the queued command, which
   `tsp -i <ID>` exposes back to `slack_notify.sh`. Single-slot daemon
   (`tsp -S 1`) guarantees one heavy job at a time.
3. **Configurable log location** — `beast_testing.sh` honours
   `BEAST_LOG_DIR` (default `$HOME/beast_logs`) and tees full stdout/stderr
   to `$BEAST_LOG_DIR/<jobname>_<UTC-ts>_<pid>.log`, plus a `latest.log`
   symlink for `tail -f`.

## Important: rotate the leaked Slack token

The previous `slack_notify.sh` had `xoxb-3361433436688-…` hard-coded and was
shared in chat. Rotate it on `api.slack.com → OAuth & Permissions → Rotate
tokens`, write the new one into `/home/aisco/.slack_token`, and never paste it
back into the script.
