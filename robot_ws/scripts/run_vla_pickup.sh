#!/usr/bin/env bash
# Create a verified target and plan, or explicitly execute, one FR5 pickup.
set -Eeuo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TARGET_FILE="/tmp/fr5_vla_target.json"
AUDIT_FILE="/tmp/fr5_vla_target_audit.png"
MAX_FRAME_AGE_SEC=45
EXECUTE=false

usage() {
    cat <<'EOF'
Usage:
  run_vla_pickup.sh "pick up the grey and orange box"
  run_vla_pickup.sh --execute "pick up the grey and orange box"

Without --execute, this creates a new camera target and plans the complete
pickup without moving the arm. --execute starts the experimental pickup
sequence immediately after target creation and successful motion preflight.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

if [[ "${1:-}" == "--execute" ]]; then
    EXECUTE=true
    shift
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 1 || -z "${1}" ]]; then
    usage >&2
    exit 2
fi
REQUEST="$1"

[[ -f /opt/ros/jazzy/setup.bash ]] || die "ROS Jazzy is not installed"
# ROS setup reads optional environment variables that may be unset in a fresh
# shell, so temporarily relax nounset only while importing it.
set +u
source /opt/ros/jazzy/setup.bash
cd -- "${WORKSPACE}"
source install/setup.bash
set -u

ros2 pkg prefix fr5_bringup >/dev/null \
    || die "fr5_bringup is not built; run scripts/start_vla_pickup.sh first"

curl --fail --silent --show-error --max-time 3 \
    http://127.0.0.1:8765/health >/dev/null \
    || die "Grounding DINO is not healthy; run scripts/start_vla_pickup.sh first"

# A real joint-state publisher is required to make base_link -> tcp_link.  Give
# the hardware driver time to establish its one RPC session after startup.
echo "Waiting for live FR5 joint states (up to 45 seconds)..."
timeout 45 ros2 topic echo --once /joint_states >/dev/null 2>&1 \
    || die "no live /joint_states; inspect the FR5 startup terminal"

echo "Creating a fresh target for: ${REQUEST}"
echo "Keep the scene still until the command returns."
ros2 run fr5_bringup vla_pick_target.py \
    --text "${REQUEST}" \
    --max-frame-age-sec "${MAX_FRAME_AGE_SEC}"

[[ -s "${TARGET_FILE}" && -s "${AUDIT_FILE}" ]] \
    || die "target artifacts were not written"

echo
echo "Target artifacts:"
echo "  ${AUDIT_FILE}"
echo "  ${TARGET_FILE}"
jq '{created, pixel, base_surface_xyz_m, score, transcript}' "${TARGET_FILE}"

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${AUDIT_FILE}" >/dev/null 2>&1 || true
fi

echo
echo "Planning complete pickup sequence (no motion)..."
ros2 run fr5_bringup d0_point_grab.py --target-file="${TARGET_FILE}"

if [[ "${EXECUTE}" != true ]]; then
    cat <<EOF

No motion occurred. Review ${AUDIT_FILE} and the printed pickup plan.
If the selected box and physical path are correct, rerun with:

  ${0} --execute "${REQUEST}"
EOF
    exit 0
fi

echo
echo "Preflight passed; --execute was supplied, so pickup is starting now."
ros2 run fr5_bringup d0_point_grab.py \
    --target-file="${TARGET_FILE}" \
    --execute --confirm-ungated-grab
