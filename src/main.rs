#[doc(inline)]
pub use std;

#[macro_use]
extern crate tracing;

mod codegen;
mod hir;
mod lexer;
mod linker;
mod syntax;
mod trace;
mod type_check;

use std::fs;
use std::path::PathBuf;
use std::process;
use structopt::StructOpt;
use tracing_subscriber::layer::SubscriberExt;

#[derive(StructOpt)]
struct Options {
    /// The source file to compile.
    source: PathBuf,
}

fn main() {
    let subscriber = tracing_subscriber::Registry::default().with(trace::Tree::new(2));
    let _ = tracing::subscriber::set_default(subscriber);

    match run() {
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
        Ok(()) => (),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
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
