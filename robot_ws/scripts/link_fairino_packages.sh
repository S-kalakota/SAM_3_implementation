#!/usr/bin/env bash
# Link the external Fairino packages required by this ROS overlay.
set -Eeuo pipefail

WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONNECTOR_ROOT="${FAIRINO_CONNECTOR_ROOT:-${HOME}/fairino_ros_connector}"
LIB_ROOT="${CONNECTOR_ROOT}/fairino_ros_libs"

die() {
    echo "error: $*" >&2
    exit 1
}

link_package() {
    local package="$1"
    local source_path="${LIB_ROOT}/${package}"
    local target_path="${WORKSPACE}/src/${package}"

    [[ -f "${source_path}/package.xml" ]] \
        || die "Fairino package not found: ${source_path} (set FAIRINO_CONNECTOR_ROOT)"

    if [[ -L "${target_path}" ]]; then
        if [[ "$(readlink -f -- "${target_path}")" == "$(readlink -f -- "${source_path}")" ]]; then
            echo "${package}: already linked"
            return
        fi
        die "${target_path} points somewhere else; remove it deliberately before relinking"
    fi
    [[ ! -e "${target_path}" ]] \
        || die "${target_path} already exists and is not a symlink"

    ln -s -- "${source_path}" "${target_path}"
    echo "${package}: linked from ${source_path}"
}

link_package fairino_description
link_package fairino_hardware_v3_9_6
link_package fairino_msgs
