"""
Static tests for isolated multi-robot description sources.

These tests intentionally avoid importing ROS packages, so the namespace
contract is also checked in lightweight Python test environments.
"""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = PROJECT_ROOT / "launch" / "simulation.launch.py"


def _launch_tree():
    return ast.parse(
        LAUNCH_PATH.read_text(encoding="utf-8"),
        filename=str(LAUNCH_PATH),
    )


def _load_pure_helper(name):
    function = next(
        node
        for node in _launch_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(LAUNCH_PATH), "exec"), namespace)
    return namespace[name]


def test_every_robot_gets_a_unique_absolute_description_source():
    description_topic = _load_pure_helper("robot_description_topic")

    assert description_topic("") == "/robot_description"
    fleet = [description_topic(f"martha_{index}") for index in range(8)]
    assert len(set(fleet)) == 8
    assert fleet == [
        f"/martha_{index}/robot_description" for index in range(8)
    ]


def test_spawn_entity_uses_explicit_topic_without_global_namespace_override():
    launch_robots = next(
        node
        for node in _launch_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "launch_robots"
    )
    spawn_calls = [
        node
        for node in ast.walk(launch_robots)
        if isinstance(node, ast.Call)
        and any(
            keyword.arg == "executable"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "spawn_entity.py"
            for keyword in node.keywords
        )
    ]

    assert len(spawn_calls) == 1
    arguments = next(
        keyword.value
        for keyword in spawn_calls[0].keywords
        if keyword.arg == "arguments"
    )
    assert isinstance(arguments, ast.Name)
    assert arguments.id == "spawn_arguments"
    assert not any(keyword.arg == "namespace" for keyword in spawn_calls[0].keywords)
    spawner_name = next(
        keyword.value
        for keyword in spawn_calls[0].keywords
        if keyword.arg == "name"
    )
    assert isinstance(spawner_name, ast.JoinedStr)

    rendered = ast.unparse(launch_robots)
    assert "spawn_arguments = ['-topic', description_topic" in rendered
    assert "'-robot_namespace'" not in rendered


def test_shared_single_robot_is_namespaced_without_changing_standalone():
    namespace_for = _load_pure_helper("robot_namespace")

    assert namespace_for(0, 1, False) == ""
    assert namespace_for(0, 1, True) == "martha_0"
    assert namespace_for(0, 4, False) == "martha_0"
    assert namespace_for(3, 4, False) == "martha_3"

    shared_source = (
        PROJECT_ROOT / "martha" / "PPO" / "shared_gazebo.py"
    ).read_text(encoding="utf-8")
    assert '"force_namespaced_fleet:=true"' in shared_source


def test_force_namespaced_fleet_rejects_ambiguous_values():
    parse_boolean = _load_pure_helper("parse_launch_boolean")

    assert parse_boolean("true", "flag") is True
    assert parse_boolean("OFF", "flag") is False
    try:
        parse_boolean("truthy", "flag")
    except RuntimeError as exc:
        assert "flag must be one of" in str(exc)
    else:
        raise AssertionError("an ambiguous launch boolean must be rejected")


def test_gazebo_plugin_namespaces_use_only_targeted_node_remaps():
    xacro = (
        PROJECT_ROOT / "urdf" / "learning.xacro"
    ).read_text(encoding="utf-8")

    assert "<namespace>" not in xacro
    assert xacro.count(":__ns:=${plugin_namespace}") == 5
    assert "<argument>__ns:=${plugin_namespace}</argument>" not in xacro
    assert 'name="gazebo_ros_bumper_${robot_namespace}"' in xacro
    assert 'name="gazebo_ros_imu_${robot_namespace}"' in xacro
    assert 'name="gazebo_ros_lidar_${robot_namespace}"' in xacro
    launch_source = LAUNCH_PATH.read_text(encoding="utf-8")
    assert "'-robot_namespace'" not in launch_source


def test_training_profile_keeps_detailed_simulation_as_the_xacro_default():
    xacro = (
        PROJECT_ROOT / "urdf" / "learning.xacro"
    ).read_text(encoding="utf-8")
    launch_source = LAUNCH_PATH.read_text(encoding="utf-8")

    assert '<xacro:arg name="training_kinematic" default="false" />' in xacro
    assert '<xacro:arg name="lidar_samples" default="360" />' in xacro
    assert "libgazebo_ros_planar_move.so" in xacro
    assert '<xacro:unless value="$(arg training_kinematic)">' in xacro
    assert '<xacro:if value="$(arg training_kinematic)">' in xacro
    assert " training_kinematic:=" in launch_source
    assert " lidar_samples:=" in launch_source


def test_spawns_and_controller_plugin_loads_are_both_serial():
    launch_robots = next(
        node
        for node in _launch_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "launch_robots"
    )
    names = {
        node.id for node in ast.walk(launch_robots) if isinstance(node, ast.Name)
    }
    strings = {
        node.value
        for node in ast.walk(launch_robots)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "startup_unpauser" not in names
    assert "gz world -p 0" not in strings

    exit_events = [
        node
        for node in ast.walk(launch_robots)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OnProcessExit"
    ]

    def keyword(call, name):
        return next(item.value for item in call.keywords if item.arg == name)

    serial_events = [
        event
        for event in exit_events
        if isinstance(keyword(event, "target_action"), ast.Name)
        and keyword(event, "target_action").id == "spawner"
    ]
    assert len(serial_events) == 1
    serial_actions = keyword(serial_events[0], "on_exit")
    assert isinstance(serial_actions, ast.List) and len(serial_actions.elts) == 1
    next_spawner = serial_actions.elts[0]
    assert isinstance(next_spawner, ast.Subscript)
    assert isinstance(next_spawner.value, ast.Name)
    assert next_spawner.value.id == "entity_spawners"

    first_controller_events = [
        event
        for event in exit_events
        if isinstance(keyword(event, "target_action"), ast.Subscript)
        and isinstance(keyword(event, "target_action").value, ast.Name)
        and keyword(event, "target_action").value.id == "entity_spawners"
    ]
    assert len(first_controller_events) == 1
    first_controller = keyword(first_controller_events[0], "on_exit")
    assert isinstance(first_controller, ast.List)
    assert isinstance(first_controller.elts[0], ast.Subscript)
    assert first_controller.elts[0].value.id == "controller_spawners"

    controller_serial_events = [
        event
        for event in exit_events
        if isinstance(keyword(event, "target_action"), ast.Name)
        and keyword(event, "target_action").id == "controller_spawner"
    ]
    assert len(controller_serial_events) == 1
    next_controller = keyword(controller_serial_events[0], "on_exit")
    assert isinstance(next_controller, ast.List)
    assert isinstance(next_controller.elts[0], ast.Subscript)
    assert next_controller.elts[0].value.id == "controller_spawners"
