"""FR5 driver + MoveIt bringup for Task A1 (Second_plan.md, Milestone A).

Starts: robot_state_publisher, ros2_control (mock or fairino_hardware),
joint_state_broadcaster -> fairino5_controller spawner chain, and move_group.
Optional RViz (rviz:=true; defaults to the value of `sim`).
Publishes the accepted B2 camera calibration as the static transform
`base_link -> zed_left_optical` by default.

Args:
  sim   -- 'true' (default) => mock_components hardware. 'false' => real FR5
           via fairino_hardware/FairinoHardwareInterface (RPC to 192.168.58.2).
           IMPORTANT: the controller tolerates only one RPC client - make sure
           the fairino_plan_executor docker stack is stopped before sim:=false.
  rviz  -- 'true'/'false'. Defaults to `sim`.
  publish_camera_tf -- publish the accepted B2 transform (default: true).
  camera_calibration -- path to the B2 `T_base_cam.json` file.

MoveIt plans in frame `base_link` (URDF root); the arm group is
`fairino5_v6_group` with tip link `wrist3_link`.
"""
import hashlib
import json
import math
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, LogInfo, OpaqueFunction,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


DEFAULT_CAMERA_CALIBRATION = str(
    Path.home() / 'VLA_Model_Work' / 'robot_ws' / 'calib' / 'T_base_cam.json')


def _is_true(value):
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _load_camera_transform(path_text):
    """Load and validate a passing B2 result before it enters the TF tree."""
    path = Path(path_text).expanduser()
    try:
        calibration = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f'Cannot load B2 camera calibration {path}: {exc}. Run '
            '`ros2 run fr5_bringup b2_fit_transform.py --write`, or launch '
            'with `publish_camera_tf:=false`.') from exc

    if calibration.get('schema_version') != 1:
        raise RuntimeError(f'Unsupported B2 calibration schema in {path}')
    if not calibration.get('quality', {}).get('passed', False):
        raise RuntimeError(f'Refusing to publish failed B2 calibration {path}')
    if calibration.get('parent_frame') != 'base_link':
        raise RuntimeError(f'B2 parent frame must be base_link in {path}')
    if calibration.get('child_frame') != 'zed_left_optical':
        raise RuntimeError(f'B2 child frame must be zed_left_optical in {path}')

    try:
        translation = [float(value) for value in calibration['t_m']]
        quaternion = [float(value) for value in calibration['quaternion_xyzw']]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'Invalid translation/quaternion in {path}') from exc
    if len(translation) != 3 or len(quaternion) != 4:
        raise RuntimeError(f'Invalid translation/quaternion dimensions in {path}')
    if not all(math.isfinite(value) for value in translation + quaternion):
        raise RuntimeError(f'Non-finite translation/quaternion in {path}')
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(quaternion_norm, 1.0, abs_tol=1e-6):
        raise RuntimeError(f'B2 quaternion is not normalized in {path}')

    # If the original capture is still alongside this deployment, catch a B1
    # recapture that has not yet been refit rather than publishing stale data.
    source_path = Path(calibration.get('source_capture', ''))
    expected_hash = calibration.get('source_sha256')
    if source_path.is_file() and expected_hash:
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f'B2 calibration {path} is stale: {source_path} changed. '
                'Run b2_fit_transform.py --write again.')
    return path, calibration, translation, quaternion


def _make_nodes(context, *args, **kwargs):
    sim = context.launch_configurations['sim'].strip().lower()
    use_fake = 'true' if sim == 'true' else 'false'

    moveit_config = (
        MoveItConfigsBuilder('fairino5_v6_robot', package_name='fr5_bringup')
        .robot_description(
            file_path='urdf/fr5.urdf.xacro',
            mappings={'use_fake_hardware': use_fake},
        )
        .robot_description_semantic(file_path='config/fr5.srdf')
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .joint_limits(file_path='config/joint_limits.yaml')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen', parameters=[moveit_config.robot_description],
    )

    import os
    controllers_yaml = os.path.join(
        moveit_config.package_path, 'config', 'ros2_controllers.yaml')
    cm = Node(
        package='controller_manager', executable='ros2_control_node',
        parameters=[moveit_config.robot_description, controllers_yaml],
        output='screen',
    )

    jsb_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )
    arm_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['fairino5_controller', '--controller-manager', '/controller_manager'],
    )
    after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=jsb_spawner, on_exit=[arm_spawner]))

    move_group = Node(
        package='moveit_ros_move_group', executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict()],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', os.path.join(moveit_config.package_path, 'config', 'moveit.rviz')],
        output='screen',
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    nodes = [rsp, cm, jsb_spawner, after_jsb, move_group, rviz]
    if _is_true(context.launch_configurations['publish_camera_tf']):
        calibration_path = context.launch_configurations['camera_calibration']
        path, calibration, translation, quaternion = _load_camera_transform(
            calibration_path)
        camera_tf = Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='zed_camera_static_tf', output='screen',
            arguments=[
                '--x', f'{translation[0]:.17g}',
                '--y', f'{translation[1]:.17g}',
                '--z', f'{translation[2]:.17g}',
                '--qx', f'{quaternion[0]:.17g}',
                '--qy', f'{quaternion[1]:.17g}',
                '--qz', f'{quaternion[2]:.17g}',
                '--qw', f'{quaternion[3]:.17g}',
                '--frame-id', calibration['parent_frame'],
                '--child-frame-id', calibration['child_frame'],
            ],
        )
        nodes.extend([
            LogInfo(msg=(
                f'Publishing accepted B2 calibration {path}: '
                f"RMS {calibration['quality']['rms_residual_mm']:.3f} mm")),
            camera_tf,
        ])

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true',
                              description="true => mock hardware; false => real FR5."),
        DeclareLaunchArgument('rviz', default_value=LaunchConfiguration('sim')),
        DeclareLaunchArgument(
            'publish_camera_tf', default_value='true',
            description='Publish base_link -> zed_left_optical from B2 calibration.'),
        DeclareLaunchArgument(
            'camera_calibration', default_value=DEFAULT_CAMERA_CALIBRATION,
            description='Path to accepted B2 T_base_cam.json.'),
        OpaqueFunction(function=_make_nodes),
    ])
