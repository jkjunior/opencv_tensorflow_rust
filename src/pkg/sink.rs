use serde_json::json;
use std::sync::mpsc::{self, SyncSender};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use super::tensor::BBox;
//use http_req::request::{self, Method, Request};
//use http_req::uri::Uri;
//use http_req::response::Headers;


pub struct Frame{
	pub boxes: Vec<BBox>,
	pub elapsed_time: u64,
	pub rtsp_url: String
}


pub struct ImageSink{
	tx: SyncSender<Frame>,
}
impl ImageSink{
	pub fn run() -> Self{
		let (tx, rx) = mpsc::sync_channel::<Frame>(32);
		std::thread::spawn(move||{
			for cam_frame in rx{
				let mut class_groups = HashMap::new();
				let boxes:Vec<_> = cam_frame.boxes.iter().map(|b|{
					let cls = class_groups.entry(&b.class_name).or_insert(0);
					*cls += 1;
					json!({
						"rect": [[b.x1, b.y1], [b.x2, b.y2]],
						"score": b.scr,
						"class_id": b.class_id,
						"class_name": b.class_name
					})
				}).collect();
				let frame = json!({
					"timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
					"class_groups": class_groups,
					"elapsed_time": cam_frame.elapsed_time,
					"boxes": boxes,
				});
				let cam = json!({
					"url": cam_frame.rtsp_url,
					"frame": frame
				});
				let dvr = json!({
					"age": "9009",
					"DVR": "Intelbras",
					"algo": "COCO",
					"cameras": [cam]
				});
				/*
				let url = "http://localhost:9200/dvr/_doc";
				let response = minreq::post(url).with_timeout(10).with_body(dvr.to_string())
					.with_header("Content-Type", "application/json").send().unwrap();
				println!("{}: {}", response.status_code, response.reason_phrase);
				println!("{}", response.as_str().unwrap());
				*/
				println!("{}", dvr.to_string());
			}
		});
		ImageSink {tx}
	}

	pub fn get_sender(&self) -> SyncSender<Frame>{
		self.tx.clone()
	}
}
