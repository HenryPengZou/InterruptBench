#!/usr/bin/env bash
#
# Workdir-independent wrapper to (re)launch the WebArena docker environment.
# It runs WebArena-Env-Setup scripts 02-06 in order.
#
# Usage:
#   bash WebAgent-R1/Eval/webarena_eval_env_reload.sh
#   bash WebAgent-R1/Eval/webarena_eval_env_reload.sh --no-homepage
#   bash WebAgent-R1/Eval/webarena_eval_env_reload.sh --with-load-images
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: webarena_eval_env_reload.sh [options]

Options:
  --with-load-images   Also run 01_docker_load_images.sh before 02-06 (slow; needs image archives)
  --no-homepage        Run 02-05 only (skip 06_serve_homepage.sh)
  -h, --help           Show this help

Notes:
  - This script is workdir-independent (paths are resolved relative to this file).
  - 02-05 are executed with sudo by default (Docker access is often root-required).
  - 06_serve_homepage.sh is executed without sudo (it calls sudo apt internally if needed).
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SETUP_DIR="${REPO_ROOT}/WebArena-Env-Setup"

WITH_LOAD_IMAGES=0
NO_HOMEPAGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-load-images)
      WITH_LOAD_IMAGES=1
      shift
      ;;
    --no-homepage)
      NO_HOMEPAGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${SETUP_DIR}" ]]; then
  echo "Error: cannot find WebArena-Env-Setup directory at: ${SETUP_DIR}" >&2
  exit 1
fi

need_file() {
  local f="$1"
  if [[ ! -f "${SETUP_DIR}/${f}" ]]; then
    echo "Error: missing required file: ${SETUP_DIR}/${f}" >&2
    exit 1
  fi
}

need_dir() {
  local d="$1"
  if [[ ! -d "${SETUP_DIR}/${d}" ]]; then
    echo "Error: missing required directory: ${SETUP_DIR}/${d}" >&2
    exit 1
  fi
}

need_file "00_vars.sh"
need_file "02_docker_remove_containers.sh"
need_file "03_docker_create_containers.sh"
need_file "04_docker_start_containers.sh"
need_file "05_docker_patch_containers.sh"
need_file "06_serve_homepage.sh"

need_dir "openstreetmap-templates"
need_dir "openstreetmap-website"
need_dir "wiki"
need_dir "webarena-homepage"

if [[ "${WITH_LOAD_IMAGES}" -eq 1 ]]; then
  need_file "01_docker_load_images.sh"
fi

run_in_setup_dir() {
  local desc="$1"
  shift
  echo "==> ${desc}"
  ( cd -- "${SETUP_DIR}" && "$@" )
}

run_sudo_bash_script() {
  local script="$1"
  run_in_setup_dir "Running ${script} (sudo)" sudo -E bash "./${script}"
}

run_bash_script() {
  local script="$1"
  run_in_setup_dir "Running ${script}" bash "./${script}"
}

if [[ "${WITH_LOAD_IMAGES}" -eq 1 ]]; then
  run_sudo_bash_script "01_docker_load_images.sh"
fi

# 02 may fail if containers do not exist; ignore that case.
set +e
run_sudo_bash_script "02_docker_remove_containers.sh"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  echo "Warning: 02_docker_remove_containers.sh returned ${rc}; continuing." >&2
fi

run_sudo_bash_script "03_docker_create_containers.sh"
run_sudo_bash_script "04_docker_start_containers.sh"
run_sudo_bash_script "05_docker_patch_containers.sh"

if [[ "${NO_HOMEPAGE}" -eq 1 ]]; then
  echo "==> Skipping 06_serve_homepage.sh (--no-homepage)."
  echo "Done."
  exit 0
fi

echo "==> Starting homepage server (this will keep running in foreground)."
echo "    You can stop it with Ctrl+C when you're done."
run_bash_script "06_serve_homepage.sh"

