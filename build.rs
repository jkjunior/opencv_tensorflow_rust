use std::process::Command;

fn main() {
    let output = Command::new("sh").arg("build.sh").output().expect("failed to execute process");
}