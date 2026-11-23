#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# The benchmark world (camera paths, attention features, noise) is generated
# at evaluation time from tracked code in eval/, so there are no remote
# datasets or model weights to fetch. Python packages (torch, numpy) are
# installed by environment/setup.sh.

set -e

cd /home/user

mkdir -p data/downloads
echo "Nothing to download: benchmark instances are generated from tracked code at evaluation time."
