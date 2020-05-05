use std::fmt::{self, Display, Formatter};
use std::rc::Rc;

use nom::branch::*;
use nom::bytes::complete::*;
use nom::character::complete::*;
use nom::combinator::*;
use nom::multi::*;
use nom::sequence::*;

#[macro_use]
mod macros;

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

keywords! {
    pub enum Keyword {
        Function = "fn",
        Struct = "struct",
        Enum = "enum",

        Extern = "extern",

        Type = "type",
        Const = "const",
        Let = "let",
        Mut = "mut",

        Loop = "loop",
        While = "while",
        For = "for",
        In = "in",

        Break = "break",
        Continue = "continue",
        Return = "return",

        If = "if",
        Else = "else",

        True = "true",
        False = "false",

        #[otherwise]
        Primitive(Primitive),
    }
}

keywords! {
    pub enum Primitive {
        U8 = "u8",
        U16 = "u16",
        U32 = "u32",
        U64 = "u64",
        I8 = "i8",
        I16 = "i16",
        I32 = "i32",
        I64 = "i64",
        F32 = "f32",
        F64 = "f64",
        Str = "str",
        Bool = "bool",
    }
}

symbols! {
    pub enum Symbol {
        Ellipses = "...",
        ThinArrow = "->",

        Equal = "==",
        NotEqual = "!=",
        GreaterThanEqual = ">=",
        LessThanEqual = "<=",
    }
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
        map(Keyword::parse, TokenKind::Keyword),
        map(literal, TokenKind::Literal),
        map(identifier, |_| TokenKind::Identifier),
        map(Symbol::parse, TokenKind::Symbol),
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
    let content = recognize(many0(alt((
        recognize(none_of("\"\\")),
        recognize(pair(char('\\'), anychar)),
    ))));
    recognize(tuple((char('"'), content, char('"'))))(input)
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
