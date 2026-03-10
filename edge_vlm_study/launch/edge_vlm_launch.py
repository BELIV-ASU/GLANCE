"""
ROS 2 Launch file for the Edge VLM Traffic Violation Detection system.

Launches:
  1. DataIngestionNode  – subscribes to camera / depth / semantic topics
  2. edge_vlm_node      – main inference & evaluation node

Usage:
    ros2 launch edge_vlm_study edge_vlm_launch.py

    # With custom config:
    ros2 launch edge_vlm_study edge_vlm_launch.py \
        config:=config/edge_vlm_config.yaml \
        benchmark:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ── Launch Arguments ─────────────────────────────────────────────────
    config_arg = DeclareLaunchArgument(
        "config",
        default_value="config/edge_vlm_config.yaml",
        description="Path to the YAML configuration file.",
    )

    benchmark_arg = DeclareLaunchArgument(
        "benchmark",
        default_value="false",
        description="Run in benchmark mode (no live ROS topics).",
    )

    model_arg = DeclareLaunchArgument(
        "model",
        default_value="qwen2_5_7b",
        description="Distilled model to load (qwen2_5_7b, cosmos_r1_7b, "
                    "mimo_7b, open_vla_7b, vila_7b).",
    )

    backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="tensorrt",
        description="Inference backend (tensorrt, onnxruntime, libtorch).",
    )

    # ── Nodes ────────────────────────────────────────────────────────────

    # Main edge VLM inference node
    edge_vlm_node = Node(
        package="edge_vlm_study",
        executable="edge_vlm_node",
        name="edge_vlm_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
        ],
        arguments=[
            # TODO: Pass benchmark flag and model/backend selection
            # "--benchmark" if LaunchConfiguration("benchmark") else ""
        ],
        remappings=[
            # TODO: Remap topics if needed
            # ("/camera/front/image_raw", "/my_camera/image"),
        ],
    )

    return LaunchDescription([
        config_arg,
        benchmark_arg,
        model_arg,
        backend_arg,
        edge_vlm_node,
    ])
