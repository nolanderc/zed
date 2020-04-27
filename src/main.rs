mod hir;
mod lexer;
mod syntax;

use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut source = String::new();
    io::stdin().read_to_string(&mut source)?;

    parse(&source)?;

    Ok(())
}

fn parse(input: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = lexer::tokenize(input).map_err(|e| nom::error::convert_error(input, e))?;
    let ast = syntax::parse(&tokens).map_err(|e| e.format(input))?;
    let _module = hir::Module::load(&ast)?;

    Ok(())
}

#[test]
fn sample() {
    let text = include_str!("../samples/sample.sol");
    if let Err(e) = parse(text) {
        panic!("error:\n{}", e);
    }
}

#[test]
fn hello_world() {
    let text = include_str!("../samples/hello_world.sol");
    if let Err(e) = parse(text) {
        panic!("error:\n{}", e);
    }
}
