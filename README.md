# AssistiveStretchVoice

**AssistiveStretchVoice** is a ROS 2-based voice control interface for the **Hello Robot Stretch** platform utilizing an AI inference layer for interpretation of both direct and indirect commands.

---

## Dependencies

Before running this project, make sure you have the following installed:

- **ROS 2 Humble**
- **Python 3.10+**
- **Hello Robot Stretch SDK** (`stretch_body`, `stretch_tool_share`, etc.)
- **OpenAI Python SDK**

## Setup

Clone and build the workspace:

```bash
git clone https://github.com/<yourusername>/AssistiveStretchVoice.git
cd AssistiveStretchVoice
colcon build
source install/setup.bash
export OPENAI_API_KEY="<insert your key>"
```

## Run Instructions

In another terminal run (and set robot position):

```bash
stretch_free_robot_process.py
ros2 launch stretch_nav2 navigation.launch.py map:=${HELLO_FLEET_PATH}/maps/Study2_Map.yaml
```

In the primary terminal run:

```bash
ros2 run voice_control voice_control_node
```


