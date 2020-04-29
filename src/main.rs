mod hir;
mod lexer;
mod syntax;
mod codegen;
mod linker;

use std::fs;
use std::path::PathBuf;

use structopt::StructOpt;

#[derive(StructOpt)]
struct Options {
    /// The source file to compile.
    source: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::from_args();

    let source = fs::read_to_string(&options.source)?;
    let module = analyze(&source)?;

    let object = "hello.o";
    codegen::emit_object(&module, object)?;
    linker::link(object);

    Ok(())
}

fn analyze(input: &str) -> Result<hir::Module, Box<dyn std::error::Error>> {
    let tokens = lexer::tokenize(input).map_err(|e| nom::error::convert_error(input, e))?;
    let ast = syntax::parse(&tokens).map_err(|e| e.format(input))?;
    let module = hir::Module::load(&ast)?;
    Ok(module)
}

#[test]
fn sample() {
    let text = include_str!("../samples/sample.zed");
    if let Err(e) = analyze(text) {
        panic!("error:\n{}", e);
    }
}

#[test]
fn hello_world() {
    let text = include_str!("../samples/hello_world.zed");
    if let Err(e) = analyze(text) {
        panic!("error:\n{}", e);
    }
}
