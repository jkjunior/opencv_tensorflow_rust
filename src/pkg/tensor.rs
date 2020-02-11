use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor, Operation};
use std::fs::File;
use std::collections::HashMap;
use std::error::Error;
use std::io::Read;
use std::iter::FromIterator;


#[derive(Clone, Debug)]
pub struct BBox {
	pub x1: i32,
	pub y1: i32,
	pub x2: i32,
	pub y2: i32,
	pub scr: f32,
	pub class_id: usize,
	pub class_name: String
}


pub struct TensorGraph{
	graph: Graph,
	session: Session,
	boxes: Operation,
	scores: Operation,
	classes: Operation,
	num_detections: Operation,
	labels: HashMap<usize, String>,
}
impl TensorGraph{
	pub fn new(model_path: &str, labels_path: &str) -> Self{
		let mut model = Vec::new();
		File::open(model_path).unwrap().read_to_end(&mut model);
		let mut graph = Graph::new();
		graph.import_graph_def(&model, &ImportGraphDefOptions::new()).unwrap();
		let session = Session::new(&SessionOptions::new(), &graph).unwrap();
		let boxes = graph.operation_by_name_required("detection_boxes").unwrap();
		let scores = graph.operation_by_name_required("detection_scores").unwrap();
		let classes = graph.operation_by_name_required("detection_classes").unwrap();
		let num_detections = graph.operation_by_name_required("num_detections").unwrap();
		let labels = Self::get_labels(&labels_path);
		TensorGraph{graph, session, boxes, scores, classes, num_detections, labels}
	}

	pub fn predict(&self, input_image:&[u8], width:u64, height:u64) -> Result<Vec<BBox>, Box<dyn Error>> {
		let input = Tensor::new(&[1, height, width, 3]).with_values(&input_image)?;
		let mut args = SessionRunArgs::new();
		args.add_feed(&self.graph.operation_by_name_required("image_tensor")?, 0, &input);
		let bbox = args.request_fetch(&self.boxes, 0);
		let scr = args.request_fetch(&self.scores, 0);
		let class = args.request_fetch(&self.classes, 0);
		let numd = args.request_fetch(&self.num_detections, 0);
		self.session.run(&mut args)?;
		let bbox_res: Tensor<f32> = args.fetch(bbox)?;
		let scr_res: Tensor<f32> = args.fetch(scr)?;
		let class_res: Tensor<f32> = args.fetch(class)?;
		let numd_res: Tensor<f32> = args.fetch(numd)?;

//		let bboxes = (0..dete_res[0] as usize).filter(|i| prob_res[*i] > 0.6).map(|i| {
		let bboxes = (0..numd_res[0] as usize).map(|i| {
			BBox {
				y1: (bbox_res[0+i*4] * height as f32) as i32,
				x1: (bbox_res[1+i*4] * width as f32) as i32,
				y2: (bbox_res[2+i*4] * height as f32) as i32,
				x2: (bbox_res[3+i*4] * width as f32) as i32,
				scr: scr_res[i],
				class_id: class_res[i] as usize,
				class_name: self.labels[&(class_res[i] as usize)].clone(),
			}
		}).collect::<Vec<_>>();
//		println!("BBox Length: {}, BBoxes:{:#?}", bboxes.len(), bboxes);
		Ok(bboxes)
	}

	fn get_labels(path:&str) -> HashMap<usize, String>{
		let mut csv = String::new();
		File::open(path).unwrap().read_to_string(&mut csv);
		HashMap::from_iter(csv.split('\n').map(|s| s.to_string()).enumerate())
	}
}
