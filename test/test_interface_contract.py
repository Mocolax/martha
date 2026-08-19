"""Static ROS interface contract tests that do not import ROS packages."""

import ast
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(relative_path):
    return PROJECT_ROOT / relative_path


def _load_yaml(relative_path):
    with _path(relative_path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    assert isinstance(document, dict), f"{relative_path} must contain a mapping"
    return document


def _ros_parameters(relative_path, node_name):
    document = _load_yaml(relative_path)
    node = document.get(node_name)
    if node is None:
        node = document.get(f"/**/{node_name}")
    assert node is not None, f"{relative_path} is missing {node_name}"
    parameters = node.get("ros__parameters")
    assert isinstance(parameters, dict)
    return parameters


def _parse_python(relative_path):
    path = _path(relative_path)
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _string_literals(tree):
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _declared_launch_arguments(tree):
    arguments = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "DeclareLaunchArgument":
            continue
        candidates = list(node.args)
        candidates.extend(
            keyword.value for keyword in node.keywords if keyword.arg == "name"
        )
        for candidate in candidates:
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                arguments.add(candidate.value)
                break
    return arguments


def _launch_argument_defaults(tree):
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "DeclareLaunchArgument":
            continue
        name_candidates = list(node.args[:1])
        name_candidates.extend(
            keyword.value for keyword in node.keywords if keyword.arg == "name"
        )
        name = next(
            (
                candidate.value
                for candidate in name_candidates
                if isinstance(candidate, ast.Constant)
                and isinstance(candidate.value, str)
            ),
            None,
        )
        default_node = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "default_value"
            ),
            None,
        )
        if name is not None and isinstance(default_node, ast.Constant):
            defaults[name] = default_node.value
    return defaults


def _literal_dict_values(tree, requested_key):
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or key.value != requested_key:
                continue
            try:
                values.append(ast.literal_eval(value))
            except (TypeError, ValueError):
                pass
    return values


def _console_scripts():
    tree = _parse_python("setup.py")
    entry_points = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "entry_points":
                entry_points = ast.literal_eval(keyword.value)
                break
    assert isinstance(entry_points, dict), "setup.py must declare entry_points"
    specifications = entry_points.get("console_scripts")
    assert isinstance(specifications, list)

    scripts = {}
    for specification in specifications:
        name, separator, target = specification.partition("=")
        assert separator, f"invalid console script: {specification}"
        name = name.strip()
        target = target.strip()
        assert name not in scripts, f"duplicate console script: {name}"
        scripts[name] = target
    return scripts


def test_all_launch_and_yaml_files_are_parseable_without_ros_imports():
    for launch_path in sorted(_path("launch").glob("*.launch.py")):
        ast.parse(
            launch_path.read_text(encoding="utf-8"),
            filename=str(launch_path),
        )

    for config_path in sorted(_path("config").glob("*.yaml")):
        with config_path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        assert isinstance(document, dict), f"{config_path.name} must contain a mapping"


def test_sim_and_hardware_ekfs_keep_the_same_public_frame_contract():
    simulation = _ros_parameters("config/ekf_rl_sim.yaml", "ekf_filter_node")
    hardware = _ros_parameters("config/ekf_hardware.yaml", "ekf_filter_node")

    assert simulation["use_sim_time"] is True
    assert hardware["use_sim_time"] is False
    assert simulation["publish_tf"] is True
    assert hardware["publish_tf"] is True
    for key in ("map_frame", "odom_frame", "base_link_frame", "world_frame"):
        assert simulation[key] == hardware[key]
    assert simulation["imu0"] == hardware["imu0"] == "/imu/data"
    assert simulation["odom0"] == "/mecanum_drive_controller/odometry"
    assert hardware["odom0"] == "/wheel/odometry"
    assert simulation["odom0_config"] == hardware["odom0_config"]
    assert simulation["imu0_config"] == hardware["imu0_config"]


def test_only_the_active_ekf_owns_dynamic_odom_tf():
    controller = _ros_parameters(
        "config/controllers.yaml",
        "mecanum_drive_controller",
    )
    simulation_tree = _parse_python("launch/simulation.launch.py")
    hardware_tree = _parse_python("launch/hardware.launch.py")

    assert controller["enable_odom_tf"] is False
    assert "ekf_node" in _string_literals(simulation_tree)
    assert "ekf_node" in _string_literals(hardware_tree)
    assert "odom_tf_broadcaster" not in _string_literals(simulation_tree)
    assert "odom_tf_broadcaster" not in _string_literals(hardware_tree)


def test_only_current_launch_files_remain():
    launch_files = {
        path.name for path in _path("launch").glob("*.launch.py")
    }
    assert launch_files == {
        "bringup.launch.py",
        "hardware.launch.py",
        "ppo_navigation.launch.py",
        "simulation.launch.py",
    }


def test_generated_and_obsolete_sources_are_absent():
    removed = (
        "config/SLAM_toolbox.yaml",
        "config/ekf.yaml",
        "martha/imu_serial_viewer.py",
        "martha/odom_tf_broadcaster.py",
        "martha/PPO/deep-research-report.md",
        "urdf/learning.urdf",
        "worlds/mundo.world",
        "pcb.zip",
    )

    assert all(not _path(path).exists() for path in removed)


def test_kicad_libraries_are_portable_and_only_manufacturing_zip_remains():
    table = _path("pcb/martha_circuits/fp-lib-table").read_text(
        encoding="utf-8"
    )

    assert "/home/" not in table
    assert "${KIPRJMOD}/../Pololu_Driver.pretty" in table
    assert "${KIPRJMOD}/../My_Components.pretty/My_Components.pretty" in table
    assert _path("pcb/martha_circuits/Gerbers.zip").is_file()
    assert not _path("pcb/martha_circuits/Gerbers").exists()


def test_hardware_bridge_exposes_the_common_command_and_sensor_topics():
    bridge = _ros_parameters(
        "config/hardware_bridge.yaml",
        "cmd_vel_serial_bridge",
    )

    assert bridge["use_sim_time"] is False
    assert bridge["input_topic"] == "/cmd_vel"
    assert bridge["odometry_topic"] == "/wheel/odometry"
    assert bridge["imu_topic"] == "/imu/data"
    assert bridge["joint_state_topic"] == "/joint_states"
    assert bridge["odom_frame"] == "odom"
    assert bridge["base_frame"] == "base_link"
    assert bridge["imu_frame"] == "imu_link"
    assert bridge["max_vx"] == 0.35
    assert bridge["max_vy"] == 0.35
    assert bridge["max_wz"] == 0.80
    assert bridge["serial_timeout"] > 0.0


def test_rplidar_a2m8_uses_the_common_scan_contract():
    lidar = _ros_parameters("config/rplidar_a2m8.yaml", "rplidar_node")

    assert lidar["channel_type"] == "serial"
    assert lidar["serial_port"] == "/dev/rplidar"
    assert lidar["serial_baudrate"] == 115200
    assert lidar["frame_id"] == "lidar"
    assert lidar["topic_name"] == "/scan"
    assert lidar["scan_mode"] == "Sensitivity"
    assert lidar["scan_frequency"] == 10.0
    assert lidar["angle_compensate"] is True
    assert lidar["inverted"] is False


def test_both_backends_accept_twist_commands_on_cmd_vel():
    bridge = _ros_parameters(
        "config/hardware_bridge.yaml",
        "cmd_vel_serial_bridge",
    )
    adapter_literals = _string_literals(
        _parse_python("martha/cmd_vel_to_twist_stamped.py")
    )

    assert bridge["input_topic"] == "/cmd_vel"
    assert "/cmd_vel" in adapter_literals
    assert "/mecanum_drive_controller/reference" in adapter_literals


def test_slam_configs_share_topics_and_frames_but_use_backend_clocks():
    simulation = _ros_parameters("config/SLAM_toolbox_sim.yaml", "slam_toolbox")
    hardware = _ros_parameters(
        "config/SLAM_toolbox_hardware.yaml",
        "slam_toolbox",
    )

    assert simulation["use_sim_time"] is True
    assert hardware["use_sim_time"] is False
    for key in ("odom_frame", "map_frame", "base_frame", "scan_topic", "mode"):
        assert simulation[key] == hardware[key]
    assert simulation["scan_topic"] == "/scan"
    assert hardware["min_laser_range"] >= 0.0
    assert hardware["max_laser_range"] > hardware["min_laser_range"]


def test_simulation_launch_has_the_rl_sensor_and_odometry_pipeline():
    tree = _parse_python("launch/simulation.launch.py")
    literals = _string_literals(tree)

    assert {"gui", "world", "sim_speed_factor", "robot_count"} <= (
        _declared_launch_arguments(tree)
    )
    assert {
        "joint_state_broadcaster",
        "mecanum_drive_controller",
        "cmd_vel_to_twist_stamped",
        "ekf_node",
        "ekf_rl_sim.yaml",
    } <= literals
    assert "odom_tf_broadcaster" not in literals
    assert True in _literal_dict_values(tree, "use_sim_time")
    assert False not in _literal_dict_values(tree, "use_sim_time")


def test_learning_urdf_declares_namespaced_contact_sensor():
    xacro = _path("urdf/learning.xacro").read_text(encoding="utf-8")

    assert 'name="contact_shell_collision"' in xacro
    assert 'type="contact"' in xacro
    assert "libgazebo_ros_bumper.so" in xacro
    assert "bumper_states:=contacts" in xacro
    assert (
        "base_link_fixed_joint_lump__contact_shell_collision_collision_1"
        in xacro
    )
    assert "<update_rate>100</update_rate>" in xacro
    assert "${robot_namespace}" in xacro


def test_bringup_launch_selects_backends_and_optional_mapping_tools():
    tree = _parse_python("launch/bringup.launch.py")
    literals = _string_literals(tree)
    defaults = _launch_argument_defaults(tree)

    assert {
        "mode",
        "world",
        "gui",
        "sim_speed_factor",
        "port",
        "lidar_port",
        "lidar_frame",
        "lidar_scan_mode",
        "start_lidar",
        "mapping",
        "rviz",
    } <= _declared_launch_arguments(tree)
    assert {
        "sim",
        "hardware",
        "simulation.launch.py",
        "hardware.launch.py",
        "SLAM_toolbox_sim.yaml",
        "SLAM_toolbox_hardware.yaml",
        "map.rviz",
    } <= literals
    assert defaults["mode"] == "sim"
    assert defaults["sim_speed_factor"] == "1.0"
    assert defaults["mapping"] == "false"
    assert defaults["rviz"] == "true"
    assert defaults["lidar_port"] == "/dev/rplidar"
    assert defaults["start_lidar"] == "true"


def test_ppo_navigation_launch_reuses_bringup_for_both_backends():
    tree = _parse_python("launch/ppo_navigation.launch.py")
    literals = _string_literals(tree)

    assert {
        "checkpoint",
        "mode",
        "world",
        "gui",
        "sim_speed_factor",
        "port",
        "lidar_port",
        "lidar_frame",
        "lidar_scan_mode",
        "start_lidar",
        "mapping",
        "rviz",
        "device",
    } <= _declared_launch_arguments(tree)
    assert "bringup.launch.py" in literals
    assert "ppo_policy" in literals


def test_mapping_rviz_uses_map_scan_and_goal_interfaces():
    document = _load_yaml("rviz/map.rviz")
    manager = document["Visualization Manager"]
    displays = {
        display["Class"]: display
        for display in manager["Displays"]
        if isinstance(display, dict) and "Class" in display
    }
    tools = {
        tool["Class"]: tool
        for tool in manager["Tools"]
        if isinstance(tool, dict) and "Class" in tool
    }

    assert manager["Global Options"]["Fixed Frame"] == "map"
    assert displays["rviz_default_plugins/Map"]["Topic"]["Value"] == "/map"
    assert displays["rviz_default_plugins/LaserScan"]["Topic"]["Value"] == "/scan"
    assert displays["rviz_default_plugins/Pose"]["Topic"]["Value"] == "/goal_pose"
    assert tools["rviz_default_plugins/SetGoal"]["Topic"] == "/goal_pose"


def test_odometry_rviz_uses_common_scan_odometry_and_goal_interfaces():
    document = _load_yaml("rviz/lidar.rviz")
    manager = document["Visualization Manager"]
    displays = {
        display["Class"]: display
        for display in manager["Displays"]
        if isinstance(display, dict) and "Class" in display
    }
    tools = {
        tool["Class"]: tool
        for tool in manager["Tools"]
        if isinstance(tool, dict) and "Class" in tool
    }

    assert manager["Global Options"]["Fixed Frame"] == "odom"
    assert displays["rviz_default_plugins/LaserScan"]["Topic"]["Value"] == "/scan"
    assert displays["rviz_default_plugins/Odometry"]["Topic"]["Value"] == (
        "/odometry/filtered"
    )
    assert displays["rviz_default_plugins/Pose"]["Topic"]["Value"] == (
        "/goal_pose"
    )
    assert tools["rviz_default_plugins/SetGoal"]["Topic"] == "/goal_pose"


def test_console_script_targets_exist_without_importing_them():
    scripts = _console_scripts()
    required = {
        "cmd_vel_to_twist_stamped",
        "cmd_vel_serial_bridge",
        "ppo_train",
        "ppo_evaluate",
        "ppo_plot",
        "ppo_policy",
    }
    assert required == scripts.keys()

    for name, target in scripts.items():
        module_name, separator, function_name = target.partition(":")
        assert separator, f"console script {name} must target module:function"
        module_path = _path(module_name.replace(".", "/") + ".py")
        assert module_path.is_file(), f"console script {name} targets a missing module"
        module_tree = ast.parse(
            module_path.read_text(encoding="utf-8"),
            filename=str(module_path),
        )
        functions = {
            node.name
            for node in module_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert function_name in functions, (
            f"console script {name} targets missing function {target}"
        )
