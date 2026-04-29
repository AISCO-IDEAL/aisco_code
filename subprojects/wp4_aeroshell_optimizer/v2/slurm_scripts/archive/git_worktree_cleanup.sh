#!/bin/bash
set -euo pipefail

# Skip the main worktree (repo root). Prompt for each auxiliary worktree.
repo_root="$(git rev-parse --show-toplevel)"
current_dir="$(pwd)"

# Parse worktrees via porcelain output for robust field handling.
while IFS=$'\t' read -r wt branch; do
  if [ "$wt" = "$repo_root" ]; then
    continue
  fi

  echo "Worktree: $wt"
  if [ -n "$branch" ]; then
    echo "Branch:   $branch"
  else
    echo "Branch:   (detached or unknown)"
  fi

  read -r -p "Delete this worktree? [y/N] " reply </dev/tty
  case "$reply" in
    y|Y)
      echo "Removing worktree: $wt"
      git worktree remove "$wt" --force

      # Delete branch if it is a local branch (refs/heads/*)
      if [[ "$branch" == refs/heads/* ]]; then
        branch_name="${branch#refs/heads/}"
        echo "Deleting branch: $branch_name"
        git branch -D "$branch_name" || true
      fi
      ;;
    *)
      echo "Skipping."
      ;;
  esac

  echo
done < <(git worktree list --porcelain | awk '
  /^worktree / { wt=$2; branch=""; next }
  /^branch /   { branch=$2; next }
  /^$/ {
    if (wt != "") {
      printf "%s\t%s\n", wt, branch
    }
    wt=""; branch=""
  }
  END {
    if (wt != "") {
      printf "%s\t%s\n", wt, branch
    }
  }
  ')

cd "$current_dir"
