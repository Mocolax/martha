import ast
import operator
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
XACRO_NS = "http://www.ros.org/wiki/xacro"

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_arithmetic(expression: str, values: dict[str, float]) -> float:
    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise AssertionError(f"unsupported xacro arithmetic: {ast.dump(node)}")

    return evaluate(ast.parse(expression, mode="eval"))


def _scalar(expression: str, values: dict[str, float]) -> float:
    expression = expression.strip()
    if expression.startswith("${") and expression.endswith("}"):
        expression = expression[2:-1]
    return _evaluate_arithmetic(expression, values)


def _xyz(expression: str, values: dict[str, float]) -> tuple[float, float, float]:
    components = re.findall(r"\$\{[^}]+\}|[^\s]+", expression)
    assert len(components) == 3
    return tuple(_scalar(component, values) for component in components)


def _expanded_properties(root: ET.Element) -> dict[str, float]:
    values: dict[str, float] = {}
    property_tag = f"{{{XACRO_NS}}}property"
    for element in root.iter(property_tag):
        expression = element.attrib["value"]
        if expression.strip().startswith("$(arg "):
            continue
        try:
            values[element.attrib["name"]] = _scalar(expression, values)
        except (AssertionError, KeyError, SyntaxError, TypeError):
            # This geometry test only expands scalar arithmetic. Xacro also
            # contains string-valued properties such as the ROS namespace.
            continue
    return values


def test_lidar_body_and_scan_plane_clear_contact_shell():
    root = ET.parse(ROOT / "urdf" / "learning.xacro").getroot()
    values = _expanded_properties(root)

    shell = root.find("./link[@name='base_link']/collision[@name='contact_shell_collision']")
    lidar_joint = root.find("./joint[@name='base_lidar']")
    assert shell is not None
    assert lidar_joint is not None

    shell_center_z = _xyz(shell.find("origin").attrib["xyz"], values)[2]
    shell_size = _xyz(shell.find("geometry/box").attrib["size"], values)
    shell_height = shell_size[2]
    lidar_center = _xyz(lidar_joint.find("origin").attrib["xyz"], values)
    lidar_center_z = lidar_center[2]

    shell_top = shell_center_z + shell_height / 2.0
    lidar_bottom = lidar_center_z - values["lidar_length"] / 2.0
    required_clearance = values["lidar_shell_clearance"]

    assert lidar_bottom == pytest.approx(shell_top + required_clearance)
    assert lidar_center_z > shell_top
    assert lidar_bottom > shell_top
    assert lidar_center[0] == pytest.approx(0.2325)
    assert lidar_center[0] + values["lidar_radius"] == pytest.approx(
        shell_size[0] / 2.0 - values["lidar_front_clearance"]
    )


def test_contact_shell_covers_kinematic_and_detailed_wheel_fronts():
    root = ET.parse(ROOT / "urdf" / "learning.xacro").getroot()
    values = _expanded_properties(root)
    shell = root.find(
        "./link[@name='base_link']/collision[@name='contact_shell_collision']"
    )
    assert shell is not None

    shell_half_x = _xyz(
        shell.find("geometry/box").attrib["size"], values
    )[0] / 2.0
    kinematic_wheel_front = values["wheel_pos_x"] + values["wheel_radius"]
    detailed_wheel_front = (
        values["wheel_pos_x"]
        + values["roller_mount_radius"]
        + values["roller_radius"]
    )

    assert shell_half_x > kinematic_wheel_front
    assert shell_half_x > detailed_wheel_front
    assert shell_half_x == pytest.approx(0.290)
