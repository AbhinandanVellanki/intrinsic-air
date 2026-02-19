#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


import numpy as np


from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration
from std_msgs.msg import Header


class WaveArm(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("WaveArm.__init__()")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"WaveArm.insert_cable() enter. Task: {task}")
        start_time = self.time_now()
        timeout = Duration(seconds=10.0)
        send_feedback("waving the arm around")
        while (self.time_now() - start_time) < timeout:
            self.sleep_for(0.25)
            observation = get_observation()
            t = (
                observation.center_image.header.stamp.sec
                + observation.center_image.header.stamp.nanosec / 1e9
            )
            self.get_logger().info(f"observation time: {t}")

            # Move the arm along a line, while looking down at the task board.
            loop_duration = 5.0  # seconds
            loop_fraction = (t % loop_duration) / loop_duration
            y_scale = 2 * loop_fraction
            if y_scale > 1.0:
                y_scale = 2.0 - y_scale
            y_scale -= 1.0

            # create a smooth series of target points that flies over the task board
            try:
                move_robot(
                    MotionUpdate(
                        header=Header(
                            frame_id="base_link",
                            stamp=self._parent_node.get_clock().now().to_msg(),
                        ),
                        pose=Pose(
                            position=Point(x=-0.4, y=0.45 + 0.3 * y_scale, z=0.25),
                            orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
                        ),
                        target_stiffness=np.diag(
                            [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
                        ).flatten(),
                        target_damping=np.diag(
                            [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
                        ).flatten(),
                        feedforward_wrench_at_tip=Wrench(
                            force=Vector3(x=0.0, y=0.0, z=0.0),
                            torque=Vector3(x=0.0, y=0.0, z=0.0),
                        ),
                        wrench_feedback_gains_at_tip=Wrench(
                            force=Vector3(x=0.5, y=0.5, z=0.5),
                            torque=Vector3(x=0.0, y=0.0, z=0.0),
                        ),
                        trajectory_generation_mode=TrajectoryGenerationMode(
                            mode=TrajectoryGenerationMode.MODE_POSITION,
                        ),
                    )
                )
            except Exception as ex:
                self.get_logger().info(f"move_robot exception: {ex}")

        self.get_logger().info("WaveArm.insert_cable() exiting...")
        return True
