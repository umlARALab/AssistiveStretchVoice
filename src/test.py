import stretch_body.robot
robot = stretch_body.robot.Robot()
robot.startup()

robot.stow()

robot.arm.move_to(0.25)
robot.push_command()

robot.lift.move_to(2)
robot.push_command()

robot.pretty_print()
robot.lift.pretty_print()

robot.head.pose('tool')
robot.head.pose('ahead')

robot.end_of_arm.move_to('wrist_yaw', 90)

robot.end_of_arm.move_to('stretch_gripper', 50)
robot.end_of_arm.move_to('stretch_gripper', -50)

robot.stow()
robot.stop()
