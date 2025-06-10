import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import speech_recognition as sr
import pyttsx3
import threading
import time
import os
import pyaudio
import sys
from contextlib import contextmanager
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
from std_msgs.msg import String
from queue import Queue
from threading import Lock
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
from geometry_msgs.msg import PoseWithCovarianceStamped
from math import sqrt
from gtts import gTTS
import pygame
import tempfile
from openai import OpenAI
import playsound
import os

@contextmanager
def ignore_stderr():
    devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            os.dup2(stderr, 2)
            os.close(stderr)
    finally:
        if devnull is not None:
            os.close(devnull)

def get_respeaker_device_id():
    with ignore_stderr():
        p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    device_id = -1
    for i in range(num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            if "ReSpeaker" in p.get_device_info_by_host_api_device_index(0, i).get('name'):
                device_id = i

    return device_id

class VoiceControlNode(Node):
    def __init__(self):
        super().__init__('voice_control_node')
        self.publisher_ = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.map_pub = self.create_publisher(String, '/map_server/map', 10)

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.update_current_pose,
            10
        )
        self.current_pose = None  
        self.goal_threshold = 0.2  

        #Drive engine
        self.engine = pyttsx3.init()
        self.engine_lock = Lock()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(get_respeaker_device_id())
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.twist = Twist()
        self.twist.angular.z = 0.0
        self.twist.linear.x = 0.0
        self.bridge = CvBridge()
        self.get_logger().info("Voice control node has been started.")

        #Command engine
        self.task_queue = Queue()
        self.current_task = None
        self.is_navigating = False
        self.current_goal_handle = None
        self.last_position = (0.0, 0.0)
        self.destination = None

        #Preset Locations
        self.kitchen = [-0.025, 2.2]
        self.livingroom = [-1.85, 0.306]
        self.bedroom = [1.49, -0.946]

        #Sound engine
        pygame.mixer.init()
        self.music_file = '/home/hello-robot/Downloads/01 - The Pink Panther Theme.mp3'  
        self.waiting_music = '/home/hello-robot/Downloads/Waiting.mp3'
        self.music_volume = 0.5  
        self.speaking = False

        try:
            pygame.mixer.music.set_volume(0)  
        except Exception as e:
            self.get_logger().error(f"Error initializing music playback: {e}")

        self.person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.current_frame = None
        self.person_detected = False

        threading.Thread(target=self.listen_for_commands, daemon=True).start()
        threading.Thread(target=self.process_task_queue, daemon=True).start()
        threading.Thread(target=self.publish_cmd_vel, daemon=True).start()

        self.load_map()

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('NavigateToPose action server not available.')

    #Load map from default file
    def load_map(self):
        map_path = os.getenv('HELLO_FLEET_PATH', '') + '/maps/Study2_Map.yaml'
        if not os.path.exists(map_path):
            self.get_logger().error(f"Map file not found at: {map_path}")
            return

        self.get_logger().info(f"Loading map from: {map_path}")
        map_msg = String()
        map_msg.data = map_path
        self.map_pub.publish(map_msg)
        time.sleep(1)

    #Person follow image callback
    def image_callback(self, msg):
        """Callback to process camera feed."""
        try:
            self.current_frame = cv2.rotate(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'),cv2.ROTATE_90_CLOCKWISE)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image message to OpenCV format: {e}")

    #Base navigation function
    def publish_cmd_vel(self):
        while rclpy.ok():
            self.publisher_.publish(self.twist)
            time.sleep(0.1)

    #Voice input and processing
    def listen_for_commands(self):
        with self.microphone as source:
            self.get_logger().info("Listening for commands...")
            self.recognizer.adjust_for_ambient_noise(source)
            while True:
                if self.speaking:
                    self.get_logger().info("Currently speaking.")
                else:
                    audio = self.recognizer.listen(source)
                    try:
                        command = self.recognizer.recognize_google(audio).lower()
                        self.get_logger().info(f"Recognized command: {command}")
                        self.handle_command(command)
                    except sr.UnknownValueError:
                        self.get_logger().info("Could not understand the command.")
                    except sr.RequestError as e:
                        self.get_logger().error(f"Speech recognition error: {e}")

    #Command processing and handling
    def handle_command(self, command):
        subcommands = self.command_interpretation(command)
        subcommands = subcommands.split(" then ")
        for subcommand in subcommands:
            subcommand = subcommand.strip()  
            if "stop" in subcommand:
                self.stop_all_tasks()
            elif "go" in subcommand and "kitchen" in subcommand:
                self.add_task_to_queue(("go", "kitchen"))
            elif "go" in subcommand and "livingroom" in subcommand:
                self.add_task_to_queue(("go", "livingroom"))
            elif "go" in subcommand and "bedroom" in subcommand:
                self.add_task_to_queue(("go", "bedroom"))
            elif "wait" in subcommand and "kitchen" in subcommand:
                self.add_task_to_queue(("wait", "kitchen"))
            elif "wait" in subcommand and "livingroom" in subcommand:
                self.add_task_to_queue(("wait", "livingroom"))
            elif "wait" in subcommand and "bedroom" in subcommand:
                self.add_task_to_queue(("wait", "bedroom"))
            elif "follow" in subcommand and "kitchen" in subcommand:
                self.add_task_to_queue(("follow", "kitchen"))
            elif "follow" in subcommand and "livingroom" in subcommand:
                self.add_task_to_queue(("follow", "livingroom"))
            elif "follow" in subcommand and "bedroom" in subcommand:
                self.add_task_to_queue(("follow", "bedroom"))
            else:
                self.speak("I dont think I got that, could you say that again?")

    #Voice command interpretation
    def command_interpretation(self, command):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "developer",
                  "content": "Your job is to interpret any input as one of these commands: go,wait,follow . And to one of these locations: kitchen,livingroom,bedroom . Always format your response like this (the number of necessary command/locations may wary and avoid repeating locations): command1,location1 then command2,location2 then command3,location3 then etc. Assume the blanket is in the bedroom and needs to get to the livingroom, the dish is in the livingroom and needs to get to the kitchen, the cup of coffe in the kitchen and needs to get to the bedroom. Avoid using follow at all costs!"},
                {
                    "role": "user",
                    "content": command
                }
            ])
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content

    #Universal halt to all activity and queue clear
    def stop_all_tasks(self):
        self.fade_out_and_stop_music()
        self.task_queue.queue.clear()
        self.is_navigating = False
        self.current_task = None
        if self.current_goal_handle:
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher_.publish(self.twist)
        self.speak("All tasks stopped.")

    #Addition of current task to the task queue
    def add_task_to_queue(self, task):
        self.task_queue.put(task)
        self.speak("Task added to the queue.")

    #Command execution off the top of the queue
    def process_task_queue(self):
        while rclpy.ok():
            if not self.is_navigating and not self.task_queue.empty():
                task = self.task_queue.get()
                task_type, task_dest = task
                self.current_task = task_type
                self.destination = task_dest
                if task_type == "go":
                    self.navigate_to(getattr(self, task_dest, "Room not found")[0], getattr(self, task_dest, "Room not found")[1])
                elif task_type == "wait":
                    self.navigate_to(getattr(self, task_dest, "Room not found")[0], getattr(self, task_dest, "Room not found")[1])
                elif task_type == "follow":
                    self.follow_to(getattr(self, task_dest, "Room not found")[0], getattr(self, task_dest, "Room not found")[1])

    #Base navigation function for map navigation after map load
    def navigate_to(self, x, y, theta=0.0):
        self.play_music_with_fade_in(self.music_file)
        self.is_navigating = True

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = np.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = np.cos(theta / 2.0)

        self.nav_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

    #Base follow function with human detection and distance calculation
    def follow_to(self, x, y):
        self.is_navigating = True
        self.play_music_with_fade_in(self.music_file)
        self.get_logger().info(f"Following to destination at ({x}, {y})")

        follow_goal_threshold = 0.25  
        original_goal_threshold = self.goal_threshold
        self.goal_threshold = follow_goal_threshold

        timeout = 30  
        start_time = time.time()

        while self.is_navigating:
            if time.time() - start_time > timeout:
                self.fade_out_and_stop_music()
                self.get_logger().info("Timeout reached while following. Stopping follow.")
                self.speak(f"I could not reach the {self.destination} in time.")
                self.is_navigating = False
                self.stop_movement()
                if self.task_queue.empty():
                    self.speak("Any further commands?")
                break

            if self.reached_goal(x, y):
                self.fade_out_and_stop_music()
                self.get_logger().info(f"Reached the destination at ({x}, {y}) while following.")
                self.speak(f"I have reached the {self.destination}.")
                self.is_navigating = False
                self.stop_movement()
                break

            person_detected, person_position = self.detect_person()
            if person_detected:
                self.adjust_velocity_to_follow(person_position)
                self.twist.angular.z = 0.0  
            else:
                self.get_logger().info("Lost track of the person. Stopping and scanning.")
                self.stop_movement()
                self.perform_camera_scan() 

            time.sleep(0.1)

        self.goal_threshold = original_goal_threshold

    #Perofrm idle rotation in a 90 degree arch to locate person
    def perform_camera_scan(self):
        self.speak("Lost track of person, performing scan.")
        self.play_music_with_fade_in(self.waiting_music)
        scan_speed = 0.2
        scan_range = math.pi / 4  
        scan_duration = scan_range / scan_speed 

        start_time = time.time()
        direction = 1

        point = 0
        prevpoint = -1

        while time.time() - start_time < scan_duration * 2:
            self.twist.angular.z = direction * scan_speed
            self.publisher_.publish(self.twist)

            if time.time() - start_time >= scan_duration:
                if point == 0 and prevpoint == -1:
                    point = 1
                    prevpoint = 0
                    start_time = time.time()
                elif point == 0 and prevpoint == 1:
                    point = -1
                    prevpoint = 0
                    start_time = time.time()
                elif point == 1 and prevpoint == 0:
                    point = 0
                    prevpoint = 1
                    start_time = time.time()
                elif point == -1 and prevpoint == 0:
                    point = 0
                    prevpoint = -1
                    start_time = time.time()

                if point == -1 or point == 1:
                    direction *= -1
                    start_time = time.time()

            person_detected, person_position = self.detect_person()
            if person_detected:
                self.get_logger().info("Person detected during scan. Aligning camera forward.")
                self.fade_out_and_stop_music()
                self.twist.angular.z = 0.0
                self.publisher_.publish(self.twist)
                return

            time.sleep(0.1)

        self.twist.angular.z = 0.0
        self.publisher_.publish(self.twist)

    #OpenCV upper body person detection 
    def detect_person(self):
        if self.current_frame is None:
            return False, None

        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)

        upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

        persons = upper_body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(100, 100)
        )

        largest_person = None
        max_area = 0
        for (x, y, w, h) in persons:
            area = w * h
            if area > max_area:
                max_area = area
                largest_person = (x, y, w, h)

        if largest_person:
            x, y, w, h = largest_person
            center_x = x + w / 2
            angular_offset = (center_x - self.current_frame.shape[1] / 2) / self.current_frame.shape[1]
            return True, {"angular_offset": angular_offset, "width": w, "height": h}

        return False, None

    #Follow person based on frame information, goal is to increase bounding box size of the detected person to a certain value
    def adjust_velocity_to_follow(self, person_position):
        desired_box_size = 900  
        box_width = person_position["width"]
        box_height = person_position["height"]
        angular_offset = person_position["angular_offset"]

        center_threshold = 0.05

        if box_width < desired_box_size or box_height < desired_box_size:
            self.get_logger().info("Box too small, moving closer.")
            self.twist.linear.x = 0.5  
        else:
            self.twist.linear.x = 0.0  

        if abs(angular_offset) > center_threshold:
            self.twist.angular.z = angular_offset * -15.0  
            self.get_logger().info(f"Turning: angular offset {angular_offset}")
        else:
            self.twist.angular.z = 0.0  

        self.publisher_.publish(self.twist)

    #Movement halt
    def stop_movement(self):
        self.fade_out_and_stop_music()
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher_.publish(self.twist)

    #Updating current position based on odometry(currentley does not use map data)
    def update_current_pose(self, msg):
        self.current_pose = msg.pose.pose

    #Check for reached goal via odometry current pose value
    def reached_goal(self, x, y):
        if self.current_pose is None:
            return False

        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        distance_to_goal = sqrt((current_x - x) ** 2 + (current_y - y) ** 2)
        self.get_logger().info(f"Distance to goal: {distance_to_goal:.2f} meters")

        return distance_to_goal < self.goal_threshold

        return distance_to_goal < self.goal_threshold
    
    #Pygame music enabler with fade in
    def play_music_with_fade_in(self, music_file):
        try:
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.set_volume(0) 
            pygame.mixer.music.play(-1)  
            for volume in range(0, int(self.music_volume * 100) + 1, 5):
                pygame.mixer.music.set_volume(volume / 100.0)
                time.sleep(0.1) 
        except Exception as e:
            self.get_logger().error(f"Error during music fade-in for {music_file}: {e}")

    #Pygame muisc half and fadeout
    def fade_out_and_stop_music(self):
        try:
            for volume in range(int(self.music_volume * 100), -1, -5):
                pygame.mixer.music.set_volume(volume / 100.0)
                time.sleep(0.1)  
            pygame.mixer.music.stop()
        except Exception as e:
            self.get_logger().error(f"Error during music fade-out: {e}")

    #Map navigation goal callback regarding if goal is reachable
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected.')
            self.speak("Navigation goal was rejected.")
            self.is_navigating = False
            return

        self.current_goal_handle = goal_handle
        self.get_logger().info('Goal accepted, waiting for result...')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    #Map navigation goal result callback regarding if the robot reached the goal or not
    def get_result_callback(self, future):
        result = future.result()
        if result is not None and hasattr(result, 'status') and result.status == 4:
            self.fade_out_and_stop_music()
            self.get_logger().info('Navigation goal succeeded!')
            self.speak(f"I have reached the {self.destination}.")
            if self.current_task == "wait":
                self.play_music_with_fade_in(self.waiting_music)
                self.speak("Waiting for 15 seconds")
                time.sleep(15)
                self.get_logger().info("Finished waiting at the destination.")
                self.fade_out_and_stop_music()
                self.speak(f"Finished waiting.")
            if self.task_queue.empty():
                self.speak("Any further commands?")
        else:
            self.get_logger().info('Navigation goal failed.')
            self.speak(f"I could not reach the {self.destination}.")
        self.is_navigating = False
        self.current_goal_handle = None

    #Audio output enabler
    def speak(self, text):
        self.speaking = True
        self.get_logger().info(f'Speaking: {text}')
        self.text_to_speech(text)
        time.sleep(0.2)
        self.speaking = False

    #Audio feedback enabler
    def text_to_speech(self, text):
        try:
            tts = gTTS(text)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
                temp_audio_path = temp_audio_file.name
                tts.save(temp_audio_path)
            playsound.playsound(temp_audio_path)
            os.remove(temp_audio_path)

        except Exception as e:
            self.get_logger().error(f'Error during TTS or playback: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = VoiceControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
