use std::env;
use std::fs;
use mlisp::interpreter::run_interpreter;
use mlisp::eval::EvalResult;

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() > 1, "Must supply file path");
    let content = fs::read_to_string(&args[1])
        .expect("Error reading file");
    if let EvalResult::Err(err) = run_interpreter(&content) {
        println!("{}", err);
    }
}
