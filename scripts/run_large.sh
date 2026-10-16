#!/usr/bin/env bash
# The whole confirmatory experiment on large datasets, one command. Resumable: run it again after a stop.
exec bash "$(dirname "$0")/setting.sh" large all
