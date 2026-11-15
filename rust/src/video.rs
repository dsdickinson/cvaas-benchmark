use std::collections::HashMap;
use std::fs;
use opencv::{
    core::{AlgorithmHint, Point, Rect, Scalar, Size},
    imgproc,
    prelude::*,
    Result,
};

pub fn load_labels(path: &str) -> HashMap<i32, String> {
    let mut labels = HashMap::new();
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return labels,
    };

    let items = content.split("item {");
    for item in items {
        if item.trim().is_empty() {
            continue;
        }
        let mut id = None;
        let mut name = None;

        for line in item.lines() {
            let line = line.trim();
            if line.starts_with("id:") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() > 1 {
                    id = parts[1].trim().parse::<i32>().ok();
                }
            } else if line.starts_with("display_name:") || line.starts_with("name:") {
                if let Some(start) = line.find('"') {
                    if let Some(end) = line.rfind('"') {
                        if start < end {
                            name = Some(line[start + 1..end].to_string());
                        }
                    }
                }
            }
        }

        if let (Some(id_val), Some(name_val)) = (id, name) {
            labels.insert(id_val, name_val);
        }
    }

    labels
}

pub fn preprocess_frame(frame: &Mat, target_width: i32, target_height: i32) -> Result<Mat> {
    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        Size::new(target_width, target_height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut rgb = Mat::default();
    imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
    Ok(rgb)
}

pub fn draw_detections(
    frame: &mut Mat,
    boxes: &[[f32; 4]],
    classes: &[i32],
    scores: &[f32],
    labels: &HashMap<i32, String>,
    min_score_thresh: f32,
) -> Result<()> {
    let size = frame.size()?;
    let h = size.height as f32;
    let w = size.width as f32;

    for i in 0..boxes.len() {
        let score = if i < scores.len() { scores[i] } else { 0.0 };
        if score < min_score_thresh {
            continue;
        }

        let cls_id = if i < classes.len() { classes[i] } else { 0 };
        let default_name = format!("class_{}", cls_id);
        let label_name = labels.get(&cls_id).unwrap_or(&default_name);
        let caption = format!("{}: {:.2}", label_name, score);

        let box_coords = boxes[i];
        let ymin = box_coords[0];
        let xmin = box_coords[1];
        let ymax = box_coords[2];
        let xmax = box_coords[3];

        let rect_x = (xmin * w).max(0.0) as i32;
        let rect_y = (ymin * h).max(0.0) as i32;
        let rect_w = ((xmax - xmin) * w).max(1.0) as i32;
        let rect_h = ((ymax - ymin) * h).max(1.0) as i32;

        let rect = Rect::new(rect_x, rect_y, rect_w, rect_h);
        imgproc::rectangle(frame, rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;

        imgproc::put_text(
            frame,
            &caption,
            Point::new(rect_x + 2, (rect_y - 4).max(12)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }

    Ok(())
}
