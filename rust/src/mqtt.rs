use serde_json::json;

pub struct MqttPublisher {
    pub enabled: bool,
    pub broker: String,
    pub topic_prefix: String,
}

impl MqttPublisher {
    pub fn new(broker: String, topic_prefix: String, enabled: bool) -> Self {
        Self {
            enabled,
            broker,
            topic_prefix,
        }
    }

    pub fn publish_detections(
        &self,
        stream_id: &str,
        boxes: &[[f32; 4]],
        classes: &[i32],
        scores: &[f32],
        timestamp: f64,
    ) {
        if !self.enabled {
            return;
        }

        let _payload = json!({
            "stream_id": stream_id,
            "timestamp": timestamp,
            "boxes": boxes,
            "classes": classes,
            "scores": scores,
        });
        // Telemetry ready
    }
}
