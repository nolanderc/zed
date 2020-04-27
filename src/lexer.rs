use std::fmt::{self, Display, Formatter};
use std::rc::Rc;

use nom::branch::*;
use nom::bytes::complete::*;
use nom::character::complete::*;
use nom::combinator::*;
use nom::error::context;
use nom::sequence::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub span: Span,
    pub text: Rc<str>,
    pub kind: TokenKind,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub length: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Identifier,
    Symbol(Symbol),
    Keyword(Keyword),
    Literal(Literal),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Keyword {
    Function,
    Struct,
    Enum,
    Trait,
    Macro,

    Type,
    Const,
    Let,
    For,
    In,

    Primitive(Primitive),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Primitive {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Str,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Symbol {
    Char(char),

    Range,
    ThinArrow,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    String,
    Integer,
    Float,
}

type Error<'a> = nom::error::VerboseError<&'a str>;
type PResult<'a, T> = nom::IResult<&'a str, T, Error<'a>>;

pub fn tokenize(input: &str) -> Result<Vec<Token>, Error> {
    let mut text = input;
    let mut tokens = Vec::new();

    loop {
        text = text.trim_start();
        if text.is_empty() {
            break;
        } else if text.starts_with("//") {
            text = text.trim_start_matches(|ch| ch != '\n');
        } else {
            match complete(token_kind)(text) {
                Ok((rest, kind)) => {
                    let token_start = input.len() - text.len();
                    let token_end = input.len() - rest.len();
                    tokens.push(Token {
                        span: Span {
                            start: token_start,
                            length: token_end - token_start,
                        },
                        text: input[token_start..token_end].into(),
                        kind,
                    });
                    text = rest;
                }
                Err(e) => match e {
                    nom::Err::Incomplete(_) => unreachable!("using complete parser"),
                    nom::Err::Error(e) | nom::Err::Failure(e) => return Err(e),
                },
            }
        }
    }

    Ok(tokens)
}

fn token_kind(input: &str) -> PResult<TokenKind> {
    alt((
        map(keyword, TokenKind::Keyword),
        map(literal, TokenKind::Literal),
        map(identifier, |_| TokenKind::Identifier),
        map(symbol, TokenKind::Symbol),
    ))(input)
}

fn keyword(input: &str) -> PResult<Keyword> {
    alt((
        map(tag("fn"), |_| Keyword::Function),
        map(tag("struct"), |_| Keyword::Struct),
        map(tag("enum"), |_| Keyword::Enum),
        map(tag("trait"), |_| Keyword::Trait),
        map(tag("macro"), |_| Keyword::Macro),
        map(tag("const"), |_| Keyword::Const),
        map(tag("type"), |_| Keyword::Type),
        map(tag("let"), |_| Keyword::Let),
        map(tag("for"), |_| Keyword::For),
        map(tag("in"), |_| Keyword::In),
        map(primitive, Keyword::Primitive),
    ))(input)
}

fn primitive(input: &str) -> PResult<Primitive> {
    alt((
        map(tag("u8"), |_| Primitive::U8),
        map(tag("u16"), |_| Primitive::U16),
        map(tag("u32"), |_| Primitive::U32),
        map(tag("u64"), |_| Primitive::U64),
        map(tag("i8"), |_| Primitive::I8),
        map(tag("i16"), |_| Primitive::I16),
        map(tag("i32"), |_| Primitive::I32),
        map(tag("i64"), |_| Primitive::I64),
        map(tag("f32"), |_| Primitive::F32),
        map(tag("f64"), |_| Primitive::F64),
        map(tag("str"), |_| Primitive::Str),
    ))(input)
}

fn literal(input: &str) -> PResult<Literal> {
    alt((
        map(string, |_| Literal::String),
        map(float, |_| Literal::Float),
        map(integer, |_| Literal::Integer),
    ))(input)
}

fn string(input: &str) -> PResult<&str> {
    let contents = escaped(is_not("\"\\"), '\\', one_of("\"\\nrta"));
    let pattern = tuple((char('"'), contents, char('"')));
    recognize(pattern)(input)
}

fn float(input: &str) -> PResult<&str> {
    let pattern = tuple((opt(char('-')), digit1, char('.'), digit1));
    recognize(pair(pattern, alphanumeric0))(input)
}

fn integer(input: &str) -> PResult<&str> {
    recognize(tuple((opt(char('-')), digit1, alphanumeric0)))(input)
}

fn identifier(input: &str) -> PResult<&str> {
    recognize(tuple((
        take_while1(|ch: char| ch.is_ascii_alphabetic() || ch == '_'),
        take_while(|ch: char| ch.is_ascii_alphanumeric() || ch == '_'),
    )))(input)
}

fn symbol(input: &str) -> PResult<Symbol> {
    alt((
        map(tag("->"), |_| Symbol::ThinArrow),
        map(tag(".."), |_| Symbol::Range),
        map(
            context("unexpected character", cut(one_of("(){}[]=.,;"))),
            Symbol::Char,
        ),
    ))(input)
}

impl From<char> for Symbol {
    fn from(ch: char) -> Self {
        Symbol::Char(ch)
    }
}

impl Display for TokenKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Symbol(symbol) => Display::fmt(symbol, f),
            TokenKind::Identifier => Display::fmt("an identifier", f),
            TokenKind::Keyword(keyword) => Display::fmt(keyword, f),
            TokenKind::Literal(literal) => Display::fmt(literal, f),
        }
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Symbol::Char(ch) => write!(f, "`{}`", ch),
            Symbol::Range => "`..`".fmt(f),
            Symbol::ThinArrow => "`->`".fmt(f),
        }
    }
}

impl Display for Keyword {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Keyword::Function => "`fn`".fmt(f),
            Keyword::Struct => "`struct`".fmt(f),
            Keyword::Enum => "`enum`".fmt(f),
            Keyword::Trait => "`trait`".fmt(f),
            Keyword::Macro => "`macro`".fmt(f),

            Keyword::Type => "`type`".fmt(f),
            Keyword::Const => "`const`".fmt(f),
            Keyword::Let => "`let`".fmt(f),
            Keyword::For => "`for`".fmt(f),
            Keyword::In => "`in`".fmt(f),

            Keyword::Primitive(primitive) => Display::fmt(primitive, f),
        }
    }
}

impl Primitive {
    pub fn as_str(self) -> &'static str {
        match self {
            Primitive::U8 => "u8",
            Primitive::U16 => "u16",
            Primitive::U32 => "u32",
            Primitive::U64 => "u64",
            Primitive::I8 => "i8",
            Primitive::I16 => "i16",
            Primitive::I32 => "i32",
            Primitive::I64 => "i64",
            Primitive::F32 => "f32",
            Primitive::F64 => "f64",
            Primitive::Str => "str",
        }
    }
}

impl Display for Primitive {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`", self.as_str())
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Literal::String => "a string literal".fmt(f),
            Literal::Integer => "an integer literal".fmt(f),
            Literal::Float => "a float literal".fmt(f),
        }
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.text.fmt(f)
    }
}
