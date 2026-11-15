import json
import base64
import cv2 as cv
from typing import Optional, Dict, Any

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class MqttPublisher:
    def __init__(self, broker: str = "localhost", port: int = 1883, topic_prefix: str = "gg/video", enabled: bool = False):
        self.enabled = enabled and MQTT_AVAILABLE
        self.broker = broker
        self.port = port
        self.topic_prefix = topic_prefix
        self.client: Optional[mqtt.Client] = None

        if self.enabled:
            try:
                self.client = mqtt.Client()
                self.client.connect(self.broker, self.port, 60)
                self.client.loop_start()
            except Exception as e:
                print(f"Warning: Failed to connect to MQTT broker {broker}:{port}: {e}")
                self.enabled = False

    def publish_detections(self, stream_id: str, boxes: list, classes: list, scores: list, timestamp: float):
        if not self.enabled or not self.client:
            return

        payload = {
            "stream_id": stream_id,
            "timestamp": timestamp,
            "boxes": boxes,
            "classes": classes,
            "scores": scores,
        }
        topic = f"{self.topic_prefix}/inf/{stream_id}"
        self.client.publish(topic, json.dumps(payload))

    def publish_frame(self, stream_id: str, frame_bgr, timestamp: float):
        if not self.enabled or not self.client:
            return

        ret, buffer = cv.imencode(".jpg", frame_bgr)
        if not ret:
            return
        b64_data = base64.b64encode(buffer).decode("utf-8")
        payload = {
            "stream_id": stream_id,
            "timestamp": timestamp,
            "image_b64": b64_data,
        }
        topic = f"{self.topic_prefix}/raw/{stream_id}"
        self.client.publish(topic, json.dumps(payload))

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
