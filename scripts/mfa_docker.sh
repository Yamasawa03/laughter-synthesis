#!/bin/bash
# MFA Docker wrapper - replaces `mfa` command with Docker execution
# Usage: MFA_COMMAND=scripts/mfa_docker.sh python3 scripts/preprocess_jvs_phoneme.py ...
# Passes all arguments through to `mfa` inside the Docker container.
# Assumes all paths passed to mfa are under ~/laughter-synthesis/

exec docker run --rm --user "$(id -u):$(id -g)" \
  -e NUMBA_CACHE_DIR=/tmp/numba_cache \
  -e MFA_ROOT_DIR=/mfa_home \
  -e MPLCONFIGDIR=/tmp/mpl \
  -v "$HOME/mfa_home:/mfa_home" \
  -v "$HOME/laughter-synthesis:/laughter-synthesis" \
  mmcauliffe/montreal-forced-aligner:v2.2.17 \
  mfa "$@"
