#!/usr/bin/env bash
# Start the resident vision service and real FR5 ROS stack for VLA pickup.
# This script does not command an arm trajectory.
set -Eeuo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PROJECT_ROOT="$(cd -- "${WORKSPACE}/.." && pwd -P)"
DINO_ROOT="${GROUNDING_DINO_ROOT:-${PROJECT_ROOT}}"

die() {
    echo "error: $*" >&2
    exit 1
}

[[ -x "${DINO_ROOT}/sam3-dino" ]] \
    || die "Grounding DINO launcher not found: ${DINO_ROOT}/sam3-dino"
[[ -f /opt/ros/jazzy/setup.bash ]] || die "ROS Jazzy is not installed"
"${WORKSPACE}/scripts/link_fairino_packages.sh"

cd -- "${DINO_ROOT}"
if ./sam3-dino status >/dev/null 2>&1; then
    echo "Grounding DINO / SAM / Qwen is already healthy."
else
    # The RTSP service owns the ZED camera. Stop the supervisor, rather than
    # its child process, so systemd does not immediately restart the holder.
    if systemctl is-active --quiet zed-rtsp.service; then
        echo "Stopping zed-rtsp.service so Grounding DINO can own the ZED camera..."
        sudo systemctl stop zed-rtsp.service
    fi
    if fuser /dev/video0 /dev/video1 >/dev/null 2>&1; then
        fuser -v /dev/video0 /dev/video1 >&2 || true
        die "a process still owns the ZED video devices"
    fi
    echo "Starting Grounding DINO / SAM / Qwen..."
    ./sam3-dino start
fi
./sam3-dino status

# ROS setup reads optional environment variables that may be unset in a fresh
# shell, so temporarily relax nounset only while importing it.
set +u
source /opt/ros/jazzy/setup.bash
cd -- "${WORKSPACE}"
set -u
colcon build --packages-select \
    fairino_description fairino_hardware_v3_9_6 fr5_bringup
# Re-source after the build so a newly installed entry point is available.
set +u
source install/setup.bash
set -u

# Both the legacy and v3.9.6 packages export the same plugin class.  Prefer
# the maintained v3.9.6 resource explicitly; otherwise pluginlib can select
# the legacy April build from the underlay.
V396_PREFIX="${WORKSPACE}/install/fairino_hardware_v3_9_6"
[[ -d "${V396_PREFIX}" ]] || die "v3.9.6 Fairino hardware package was not installed"
LEGACY_HARDWARE_PREFIX="${LEGACY_FAIRINO_PREFIX:-${HOME}/ros2_ws/install/fairino_hardware}"

without_path() {
    local value="$1" excluded="$2" entry result=""
    local -a entries
    IFS=':' read -r -a entries <<< "${value}"
    for entry in "${entries[@]}"; do
        [[ -z "${entry}" || "${entry}" == "${excluded}" ]] && continue
        result+="${result:+:}${entry}"
    done
    printf '%s' "${result}"
}

# Pluginlib sees two packages that export the same class name.  It otherwise
# loads the legacy April plugin even though the v3.9.6 prefix is first.
export AMENT_PREFIX_PATH="${V396_PREFIX}:$(without_path "${AMENT_PREFIX_PATH}" "${LEGACY_HARDWARE_PREFIX}")"
export LD_LIBRARY_PATH="${V396_PREFIX}/lib:$(without_path "${LD_LIBRARY_PATH:-}" "${LEGACY_HARDWARE_PREFIX}/lib")"

if pgrep -u "$(id -u)" -f \
    'ros2 launch fr5_bringup a1_bringup.launch.py.*sim:=false' >/dev/null; then
    die "the real FR5 bringup is already running; use scripts/run_vla_pickup.sh in another terminal"
fi

cat <<EOF

Grounding DINO is ready.  Starting the real FR5 stack now.
This connects to the controller but does not command a robot trajectory.
Leave this terminal running.  In another terminal, run for example:

  ${WORKSPACE}/scripts/run_vla_pickup.sh \
    "pick up the grey and orange box"
EOF

exec ros2 launch fr5_bringup a1_bringup.launch.py sim:=false
