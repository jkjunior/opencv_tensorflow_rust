#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_must_use)]
use opencv::prelude::*;
use opencv::imgcodecs;
use opencv::videoio;
use opencv::imgproc;
//use opencv::highgui;
use serde_yaml::Value;
use std::fs::File;
use std::sync::mpsc::{self, SyncSender};
//use tflite::ops::builtin::BuiltinOpResolver;
//use tflite::{FlatBufferModel, InterpreterBuilder};
use opencv::core::{Scalar, Rect, Point, FONT_HERSHEY_SIMPLEX, LINE_8, LINE_AA, add_weighted};
use std::error::Error;
use std::time::Instant;
mod pkg;
use pkg::tensor::{BBox, TensorGraph};
use pkg::sink::{ImageSink, Frame};


const COLORMAP:[(f64, f64, f64); 7] = [(0.0, 127.0, 255.0), (0.0, 255.0, 0.0), (0.0, 0.0, 255.0), (255.0, 255.0, 0.0), (255.0, 0.0, 255.0), (0.0, 255.0, 255.0), (255.0, 255.0, 255.0)];


struct CameraCap;
impl CameraCap{
	fn run(conf: &Value, writer_sender: SyncSender<Frame> ){
		let model_path = conf["model"].as_str().unwrap().to_string();
		let labels_path = conf["labels"].as_str().unwrap().to_string();
		let rtsp_url = conf["rtsp_url"].as_str().unwrap().to_string();
		let movie = conf["movie"].as_bool().unwrap();
		let tensor_graph = TensorGraph::new(&model_path, &labels_path);
		let (tx, rx) = mpsc::sync_channel::<Mat>(32);
		let cam_url = rtsp_url.clone();
//		highgui::named_window("Video Feed", highgui::WINDOW_AUTOSIZE);
		std::thread::spawn(move||{
			let gui_delay = if movie {1} else {100};
			for mut frame in rx{
				let buffer = Self::mat_bgr(&frame);
//				let buffer = Self::mat_rgb(&frame);
				let frame_size = frame.size().unwrap();
				let now = Instant::now();
				let boxes = tensor_graph.predict(&buffer, frame_size.width as u64, frame_size.height as u64).unwrap();
				Self::print_boxes(&mut frame, &boxes);
				let cam_frame = Frame{boxes, elapsed_time: now.elapsed().as_millis() as u64, rtsp_url: cam_url.clone()};
//				highgui::imshow("Video Feed", &frame);
//				opencv::highgui::wait_key(gui_delay);
				writer_sender.send(cam_frame);
			}
		});
		if movie {
			//let mut cap = videoio::VideoCapture::new(0).unwrap();
			let mut cap = videoio::VideoCapture::new_from_file_with_backend(&rtsp_url, videoio::CAP_FFMPEG).unwrap();
//			let cam_fps = cap.get(videoio::CAP_PROP_FPS).unwrap();
			let cam_fps = 30.0;
			let mut out_fps = conf["out_fps"].as_f64().unwrap();
			if out_fps > cam_fps || out_fps < std::f64::MIN_POSITIVE { out_fps = cam_fps };
			let period = (cam_fps / out_fps) as u32;
			let mut tick = 0u32;
			std::thread::spawn(move|| {
				loop {
					//cap.read(&mut frame);
					cap.grab();
					if tick % period == 0 {
						let mut frame = Mat::default().unwrap();
						cap.retrieve(&mut frame, 0);
						tx.send(frame);
					}
					tick = tick.overflowing_add(1).0;
				}
			});
		}else{
			let img = imgcodecs::imread(conf["still_image"].as_str().unwrap(), imgcodecs::IMREAD_UNCHANGED).unwrap();
			tx.send(img);
			std::thread::sleep(std::time::Duration::from_secs(std::u64::MAX));
		}
	}

	fn mat_bgr(frame: &Mat) -> Vec<u8>{
//		let buffer:Vec<Vec<u8>> = img.reshape(1, 1).unwrap().to_vec_2d().unwrap();
//		let mut buffer = buffer.into_iter().flatten().collect::<Vec<_>>();
//		let mut buffer = &buffer[0];
		let reshaped = frame.reshape(1, 1).unwrap();
		reshaped.at_row(0).unwrap().to_vec()
	}

	fn mat_rgb(frame: &Mat) -> Vec<u8>{
//		let mut rgb = Mat::default().unwrap();
//		imgproc::cvt_color(&img, &mut rgb, imgproc::COLOR_BGR2RGB, 0);
//		let buffer = rgb.reshape(1, 1)?;
//		let mut buffer:&[u8] = buffer.at_row(0)?;
		let mut bgr = Self::mat_bgr(frame);
		bgr.chunks_mut(3).for_each(|x| x.reverse());
		bgr
	}

	fn print_boxes(frame:&mut Mat, boxes:&[BBox]){
		let scale = frame.size().unwrap().height/33;
		for bbox in boxes {
			let color = COLORMAP[bbox.class_id % COLORMAP.len()];
			imgproc::rectangle(frame, Rect::new(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, scale),
							   Scalar::new(color.0, color.1, color.2, 0.0), -1, 8, 0);
			imgproc::rectangle(frame, Rect::new(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1),
							   Scalar::new(color.0, color.1, color.2, 0.0), 1, 8, 0);
			imgproc::put_text(frame, &format!("{}: {:.2}", bbox.class_name, bbox.scr), Point::new(bbox.x1, bbox.y1 + 2*scale/3), 0,
							  0.33, Scalar::new(0.0, 0.0, 0.0, 0.0), 1, 16, false);
		}
	}
}


fn main(){
//	let conf = serde_yaml::from_reader(File::open("/home/josias/workspace/IdeaProjects/rust/opencv_rtsp/rsrc/config.yml").unwrap()).unwrap();
	let conf = serde_yaml::from_reader(File::open("/home/joao/repositories/opencv_rtsp/rsrc/config.yml").unwrap()).unwrap();
	let imsink = ImageSink::run();
	CameraCap::run(&conf, imsink.get_sender());
//	CameraCap::still_image(&conf);
//	cap_cam(model);
//	test_img(model);
	std::thread::sleep(std::time::Duration::from_millis(std::u64::MAX));
}

//	let mut img = Mat::from_slice(&buffer).unwrap();
//	img = img.reshape(3, h).unwrap();
//	let mut img = imgcodecs::imdecode(&VectorOfuchar::from_iter(img), imgcodecs::IMREAD_UNCHANGED).unwrap();
//	imgcodecs::imencode(".png", &img, &mut buffer, &opencv::types::VectorOfint::new());

/*
		imgproc::resize(&src, &mut dst, Size::new(w, h), 0.0, 0.0, imgproc::INTER_LINEAR);
		dst = Mat::roi(&dst, Rect::new(dw, dh, w, h)).unwrap();

//	let mut dst = Mat::new_rows_cols_with_default(w, h, CV_8UC3, Scalar::all(255.0)).unwrap();
		Mat::roi(&src, Rect::new(dw/2, dh/2, img_size.width - dw, img_size.height - dh)).unwrap()
		let mut dst = Mat::default().unwrap();
		copy_make_border(&crop, &mut dst, dh, dh, dw, dw, BORDER_REPLICATE, Scalar::all(255.0));


	let mut buffer = Vec::new();
	File::open(path).unwrap().read_to_end(&mut buffer);
	let model_arc = Arc::new(model);
	let mut tds = (0..4).map(|_| {
		let (tx, rx) = mpsc::sync_channel::<(Mat, Vec<u8>)>(10);
		let model_ref = model_arc.clone();
		std::thread::spawn(move||{
			for (mut frame, buffer) in rx{
				let bboxes = tensor_graph(&model_ref, buffer).unwrap();
				for bbox in bboxes {
					imgproc::rectangle(&mut frame, Rect::new(bbox.x1 as i32, bbox.y1 as i32, (bbox.x2 - bbox.x1) as i32, (bbox.y2 - bbox.y1) as i32),
									   Scalar::new(0.0,255.0,0.0, 0.0), 1, 8, 0);
				}
				highgui::imshow("Video Feed", &frame);
//			imgcodecs::imwrite(&format!("/mnt/ramdisk/camera-{}.jpg", tick), &frame, &opencv::types::VectorOfint::new());
			}
		});
		tx
	}).collect::<Vec<_>>();
	let mut tds = tds.iter().cycle();
*/

//fn tensor_run<P: AsRef<Path>>(model_path: P, img:Vec<u8>) -> Result<Vec<BBox>, Box<dyn Error>> {

//https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/detect_picamera.py
//https://github.com/boncheolgu/tflite-rs
//https://www.tensorflow.org/lite

//		imgproc::rectangle(overlay, Rect::new(bbox.x1, bbox.y1 - scale, bbox.x2 - bbox.x1, scale),
//		Scalar::new(color.0, color.1, color.2, 0.0), -1, LINE_8, 0);

//		imgproc::put_text(frame, &format!("{}: {:.2}", "person", bbox.prob), Point::new(bbox.x1,bbox.y1 - scale/4), FONT_HERSHEY_SIMPLEX,
//						  0.33, Scalar::new(color.0, color.1, color.2, 0.0), 1, LINE_AA, false);

//	let alpha = 0.25;
//	let overlay = &mut frame.clone().unwrap();
//		add_weighted(overlay, alpha, &mut frame.clone().unwrap(), 1.0-alpha, 0.0, frame, -1);

/*

{
  "age": 9009,
  "algo": "COCO",
  "cameras": [
    {
      "url": "rtsp://172.16.165.151:8554",
      "frame": {
        "time_stamp": 12111334,
        "boxes": [
          {
            "rec": [
              [
                "x1",
                "y1"
              ],
              [
                "x2",
                "y2"
              ]
            ],
            "scr": 0.3,
            "class_id": 2,
            "class_name": "person"
          }
        ]
      }
    }
  ]
}

*/