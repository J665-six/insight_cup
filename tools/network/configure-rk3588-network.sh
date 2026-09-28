#!/bin/sh
set -eu

exec systemctl --user restart rk3588-network-share.service
