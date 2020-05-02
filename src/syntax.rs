use crate::lexer::*;

use thiserror::Error;

use nom::branch::*;
use nom::combinator::*;
use nom::error::{context, ErrorKind as NomErrorKind, ParseError};
use nom::multi::*;
use nom::sequence::*;

use std::error::Error as StdError;
use std::fmt::{Display, Write};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct Ast {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Type(ItemType),
    Const(ItemConst),
}

#[derive(Debug, Clone)]
pub struct ItemType {
    pub type_token: Token,
    pub ident: Ident,
    pub generics: Option<Generics>,
    pub eq_token: Token,
    pub decl: TypeDecl,
}

#[derive(Debug, Clone)]
pub struct ItemConst {
    pub const_token: Token,
    pub ident: Ident,
    pub eq_token: Token,
    pub const_init: ConstInit,
}

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub ty: Type,
    pub semi_token: Option<Token>,
}

#[derive(Debug, Clone)]
pub enum ConstInit {
    Function(Function),
    Extern(ExternFunction),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Block(Box<ExprBlock>),
    Call(Box<ExprCall>),
    Binding(Box<ExprBinding>),
    Literal(Box<ExprLiteral>),
    Loop(Box<ExprLoop>),
    While(Box<ExprWhile>),
    ForLoop(Box<ExprFor>),
    Constructor(Box<ExprConstructor>),
    Ident(Box<Ident>),
    Infix(Box<ExprInfix>),
    If(Box<ExprIf>),
    Assign(Box<ExprAssign>),
    Control(Box<ExprControl>),
}

#[derive(Debug, Clone)]
pub enum ExprControl {
    Break(ExprBreak),
    Continue(ExprContinue),
}

#[derive(Debug, Clone)]
pub struct ExprBreak {
    pub break_token: Token,
}

#[derive(Debug, Clone)]
pub struct ExprContinue {
    pub continue_token: Token,
}

#[derive(Debug, Clone)]
pub struct ExprAssign {
    pub ident: Ident,
    pub eq_token: Token,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct ExprIf {
    pub if_token: Token,
    pub condition: Expr,
    pub block: ExprBlock,
    pub else_branch: Option<ExprElse>,
}

#[derive(Debug, Clone)]
pub struct ExprElse {
    pub else_token: Token,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct ExprInfix {
    pub operator: BinaryOperator,
    pub lhs: Expr,
    pub rhs: Expr,
}

macro_rules! binary_operators {
    (
        $vis:vis enum $ident:ident {
            $(
                $op:ident ($precedence:literal) = $symbol:expr,
            )*
        }
    ) => {
        #[derive(Debug, Copy, Clone)]
        pub enum $ident {
            $($op),*
        }

        impl $ident {
            fn parse(input: Input) -> PResult<BinaryOperator> {
                alt((
                        $(
                            map(symbol($symbol), |_| $ident::$op),
                        )*
                ))(input)
            }
        }
    }
}

binary_operators! {
    pub enum BinaryOperator {
        Dot (2) = '.',
        Range (1) = Symbol::Range,
    }
}

#[derive(Debug, Clone)]
pub struct ExprBlock {
    pub open_curly: Token,
    pub sequence: Vec<ExprSeq>,
    pub close_curly: Token,
}

#[derive(Debug, Clone)]
pub struct ExprSeq {
    pub expr: Expr,
    pub semi_token: Option<Token>,
}

type Argument = Expr;

#[derive(Debug, Clone)]
pub struct ExprCall {
    pub ident: Ident,
    pub open_parens: Token,
    pub arguments: Vec<Argument>,
    pub close_parens: Token,
}

#[derive(Debug, Clone)]
pub struct ExprBinding {
    pub let_token: Token,
    pub mut_token: Option<Token>,
    pub ident: Ident,
    pub eq_token: Token,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub enum ExprLiteral {
    Bool(ExprBool),
    String(ExprString),
    Integer(ExprInteger),
    Float(ExprFloat),
}

#[derive(Debug, Clone)]
pub struct ExprBool {
    pub token: Token,
    pub value: bool,
}

#[derive(Debug, Clone)]
pub struct ExprString {
    pub token: Token,
    pub value: Rc<str>,
}

#[derive(Debug, Clone)]
pub struct ExprInteger {
    pub token: Token,
    pub value: Integer,
}

#[derive(Debug, Clone)]
pub enum Integer {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Any(Rc<str>),
}

#[derive(Debug, Clone)]
pub struct ExprFloat {
    pub token: Token,
    pub value: Float,
}

#[derive(Debug, Clone)]
pub enum Float {
    F32(f32),
    F64(f64),
}

#[derive(Debug, Clone)]
pub struct ExprFor {
    pub for_token: Token,
    pub ident: Ident,
    pub in_token: Token,
    pub range: Expr,
    pub body: ExprBlock,
}

#[derive(Debug, Clone)]
pub struct ExprLoop {
    pub loop_token: Token,
    pub body: ExprBlock,
}

#[derive(Debug, Clone)]
pub struct ExprWhile {
    pub while_token: Token,
    pub condition: Expr,
    pub body: ExprBlock,
}

#[derive(Debug, Clone)]
pub struct ExprConstructor {
    pub ident: Ident,
    pub open_curly: Token,
    pub initializers: Vec<Initializer>,
    pub close_curly: Token,
}

#[derive(Debug, Clone)]
pub struct Initializer {
    pub dot_token: Token,
    pub ident: Ident,
    pub eq_token: Token,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub arrow_token: Token,
    pub ty: Type,
}

pub type Fields = Vec<Field>;

#[derive(Debug, Clone)]
pub struct Field {
    pub ident: Ident,
    pub ty: Type,
}

pub type Variants = Vec<Variant>;

#[derive(Debug, Clone)]
pub struct Variant {
    pub ident: Ident,
    pub eq_token: Option<Token>,
    pub ty: Option<Type>,
}

pub type Ident = Token;

#[derive(Debug, Clone)]
pub enum Type {
    Primitive(PrimitiveType),
    Struct(Struct),
    Enum(Enum),
    Alias(Ident),
    Unit,
}

#[derive(Debug, Clone)]
pub struct PrimitiveType {
    pub token: Option<Token>,
    pub kind: Primitive,
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub struct_token: Token,
    pub open_curly: Token,
    pub fields: Fields,
    pub close_curly: Token,
}

#[derive(Debug, Clone)]
pub struct Enum {
    pub enum_token: Token,
    pub open_curly: Token,
    pub variants: Variants,
    pub close_curly: Token,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub fn_token: Token,
    pub signature: Signature,
    pub body: ExprBlock,
}

#[derive(Debug, Clone)]
pub struct ExternFunction {
    pub extern_token: Token,
    pub fn_token: Token,
    pub signature: Signature,
    pub semi_token: Token,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub open_parens: Token,
    pub arguments: Fields,
    pub ellipses: Option<Token>,
    pub close_parens: Token,
    pub return_clause: Option<ReturnClause>,
}

#[derive(Debug, Clone)]
pub struct Generics {
    pub lt_token: Token,
    pub idents: Vec<Ident>,
    pub gt_token: Token,
}

#[derive(Debug, Clone, Error)]
#[error("failed to parse")]
pub struct Error {
    location: SourceLocation,
    #[source]
    kind: ErrorKind,
}

#[derive(Debug, Clone, Error)]
pub enum ErrorKind {
    #[error("combinator {0:?}")]
    Nom(NomErrorKind),

    #[error("{0}")]
    Custom(&'static str),

    #[error("failed to parse integer")]
    ParseInteger(#[from] std::num::ParseIntError),
    #[error("failed to parse float")]
    ParseFloat(#[from] std::num::ParseFloatError),

    #[error("unexpected token: {0}")]
    UnexpectedToken(TokenKind),

    #[error("found {found}, expected {expected}", found = display_or(found.as_ref(), "EOF"))]
    ExpectedToken {
        expected: TokenKind,
        found: Option<TokenKind>,
    },
    #[error("expected a primitve type (`u8`, `i32`, ...)")]
    ExpectedPrimitive,

    #[error("invalid integer suffix")]
    InvalidIntegerSuffix,
    #[error("invalid float suffix")]
    InvalidFloatSuffix,
}

fn display_or<T: Display>(value: Option<T>, default: impl Into<String>) -> String {
    value
        .as_ref()
        .map(|v| v.to_string())
        .unwrap_or_else(|| default.into())
}

/// Byte offset into the source text.
type SourceLocation = Option<usize>;

type Input<'a> = &'a [Token];
type PResult<'a, T> = nom::IResult<Input<'a>, T, Error>;

impl Error {
    fn source_location(input: Input) -> SourceLocation {
        input.first().map(|token| token.span.start)
    }

    pub fn single(input: Input, kind: ErrorKind) -> Self {
        Error {
            location: Self::source_location(input),
            kind,
        }
    }

    pub fn format(&self, source: &str) -> String {
        let mut result = String::new();

        let pos = self.location.unwrap_or_else(|| source.len());
        let line_start = source[..pos]
            .rfind('\n')
            .map(|start| start + 1)
            .unwrap_or(0);
        let line_end = source[pos..]
            .find('\n')
            .map(|end| pos + end)
            .unwrap_or_else(|| source.len());
        let snippet = &source[line_start..line_end];
        let line_offset = pos - line_start;

        result.push_str("Error: ");

        let mut error: &(dyn StdError + 'static) = &self.kind;
        loop {
            write!(result, "{}: ", error).unwrap();
            match error.source() {
                Some(inner) => error = inner,
                None => {
                    if let Some(local) = error.downcast_ref::<Box<Self>>() {
                        let inner = local.format(source);
                        for line in inner.lines() {
                            result.push('\n');
                            result.push_str("    ");
                            result.push_str(line);
                        }
                        result.push('\n');
                    }
                    break;
                }
            }
        }

        writeln!(
            result,
            "\n{}\n{blank:offset$}^\n",
            snippet,
            blank = "",
            offset = line_offset
        )
        .unwrap();

        result
    }
}

impl<'a> ParseError<Input<'a>> for Error {
    fn from_error_kind(input: Input<'a>, kind: NomErrorKind) -> Self {
        Error::single(input, ErrorKind::Nom(kind))
    }

    fn append(_input: Input<'a>, _kind: NomErrorKind, other: Self) -> Self {
        other
    }
}

pub fn parse(tokens: Input) -> Result<Ast, Error> {
    let mut tokens = tokens;

    let mut items = Vec::new();

    while !tokens.is_empty() {
        match context("when parsing item", item)(tokens) {
            Ok((rest, item)) => {
                items.push(item);
                tokens = rest;
            }
            Err(e) => match e {
                nom::Err::Incomplete(_) => unreachable!("using complete parser"),
                nom::Err::Error(e) | nom::Err::Failure(e) => return Err(e),
            },
        }
    }

    Ok(Ast { items })
}

fn item(input: Input) -> PResult<Item> {
    alt((map(item_type, Item::Type), map(item_const, Item::Const)))(input)
}

fn item_type(input: Input) -> PResult<ItemType> {
    map(
        tuple((
            keyword(Keyword::Type),
            cut(tuple((ident, opt(generics), symbol('='), type_decl))),
        )),
        |(type_token, (ident, generics, eq_token, decl))| ItemType {
            type_token,
            ident,
            generics,
            eq_token,
            decl,
        },
    )(input)
}

fn type_decl(input: Input) -> PResult<TypeDecl> {
    let (remaining, ty) = parse_type(input)?;

    let (rest, semi_token) = match ty {
        Type::Alias(_) | Type::Unit | Type::Primitive(_) => map(symbol(';'), Some)(remaining)?,
        Type::Enum(_) | Type::Struct(_) => (remaining, None),
    };

    Ok((rest, TypeDecl { ty, semi_token }))
}

fn item_const(input: Input) -> PResult<ItemConst> {
    map(
        tuple((
            keyword(Keyword::Const),
            cut(tuple((ident, symbol('='), const_init))),
        )),
        |(const_token, (ident, eq_token, const_init))| ItemConst {
            const_token,
            ident,
            eq_token,
            const_init,
        },
    )(input)
}

fn const_init(input: Input) -> PResult<ConstInit> {
    alt((
        map(extern_function, ConstInit::Extern),
        map(function, ConstInit::Function),
    ))(input)
}

fn extern_function(input: Input) -> PResult<ExternFunction> {
    map(
        tuple((
            keyword(Keyword::Extern),
            cut(tuple((
                keyword(Keyword::Function),
                function_signature,
                symbol(';'),
            ))),
        )),
        |(extern_token, (fn_token, signature, semi_token))| ExternFunction {
            extern_token,
            fn_token,
            signature,
            semi_token,
        },
    )(input)
}

fn function(input: Input) -> PResult<Function> {
    map(
        tuple((
            keyword(Keyword::Function),
            cut(tuple((
                function_signature,
                context("function body", expr_block),
            ))),
        )),
        |(fn_token, (signature, body))| Function {
            fn_token,
            signature,
            body,
        },
    )(input)
}

fn function_signature(input: Input) -> PResult<Signature> {
    let fields = separated_list(symbol(','), field);

    map(
        tuple((
            symbol('('),
            fields,
            opt(preceded(symbol(','), opt(symbol(Symbol::Ellipses)))),
            symbol(')'),
            opt(return_clause),
        )),
        |(open_parens, arguments, ellipses, close_parens, return_clause)| Signature {
            open_parens,
            arguments,
            ellipses: ellipses.flatten(),
            close_parens,
            return_clause,
        },
    )(input)
}

fn return_clause(input: Input) -> PResult<ReturnClause> {
    map(
        pair(symbol(Symbol::ThinArrow), parse_type),
        |(arrow_token, ty)| ReturnClause { arrow_token, ty },
    )(input)
}

fn expr_statement(input: Input) -> PResult<Expr> {
    alt((
        map(context("let binding", expr_binding), |e| {
            Expr::Binding(e.into())
        }),
        map(context("for loop", expr_for_loop), |e| {
            Expr::ForLoop(e.into())
        }),
        map(context("while loop", expr_while), |e| Expr::While(e.into())),
        map(context("assignment", expr_assign), |e| {
            Expr::Assign(e.into())
        }),
        expr_inline,
    ))(input)
}

fn expr_primary(input: Input) -> PResult<Expr> {
    alt((
        map(context("endless loop", expr_loop), |e| Expr::Loop(e.into())),
        map(context("control flow", expr_control), |e| {
            Expr::Control(e.into())
        }),
        map(context("if", expr_if), |e| Expr::If(e.into())),
        map(context("block", expr_block), |e| Expr::Block(e.into())),
        map(context("function call", expr_call), |e| {
            Expr::Call(e.into())
        }),
        map(context("literal constant", expr_literal), |e| {
            Expr::Literal(e.into())
        }),
        map(context("constructor", expr_constructor), |e| {
            Expr::Constructor(e.into())
        }),
        map(context("ident", ident), |e| Expr::Ident(e.into())),
    ))(input)
}

fn expr_inline(input: Input) -> PResult<Expr> {
    let (rest, lhs) = expr_primary(input)?;
    expr_infix(lhs, 0)(rest)
}

fn expr_infix(mut lhs: Expr, min_precedence: u32) -> impl FnOnce(Input) -> PResult<Expr> {
    move |mut input| {
        while let (rest, Some(operator)) = opt(BinaryOperator::parse)(input)? {
            if operator.precedence() < min_precedence {
                break;
            }

            let (rest, mut rhs) = expr_primary(rest)?;
            input = rest;

            while let (rest, Some(next_operator)) = opt(BinaryOperator::parse)(input)? {
                if next_operator.precedence() > operator.precedence() {
                    let (rest, next) = expr_infix(rhs, next_operator.precedence())(rest)?;
                    input = rest;
                    rhs = next;
                } else {
                    break;
                }
            }

            input = rest;
            lhs = Expr::Infix(Box::new(ExprInfix { operator, lhs, rhs }));
        }

        Ok((input, lhs))
    }
}

impl BinaryOperator {
    fn precedence(self) -> u32 {
        match self {
            BinaryOperator::Dot => 2,
            BinaryOperator::Range => 1,
        }
    }
}

fn expr_if(input: Input) -> PResult<ExprIf> {
    map(
        tuple((
            keyword(Keyword::If),
            cut(tuple((expr_inline, expr_block, opt(expr_else)))),
        )),
        |(if_token, (condition, block, else_branch))| ExprIf {
            if_token,
            condition,
            block,
            else_branch,
        },
    )(input)
}

fn expr_else(input: Input) -> PResult<ExprElse> {
    map(
        tuple((
            keyword(Keyword::Else),
            cut(alt((
                map(expr_if, |e| Expr::If(e.into())),
                map(expr_block, |e| Expr::Block(e.into())),
            ))),
        )),
        |(else_token, expr)| ExprElse { else_token, expr },
    )(input)
}

fn expr_block(input: Input) -> PResult<ExprBlock> {
    let mut sequence = Vec::<ExprSeq>::new();

    let (mut tokens, open_curly) = symbol('{')(input)?;

    loop {
        let (remaining, expr) = match expr_statement(tokens) {
            Ok(result) => result,
            Err(e @ nom::Err::Failure(_)) => return Err(e),
            Err(_) => break,
        };

        tokens = remaining;

        let (rest, semi_token) = match expr {
            // semicolon fully optional
            Expr::Block(_) | Expr::If(_) | Expr::Loop(_) | Expr::ForLoop(_) | Expr::While(_) => {
                opt(symbol(';'))(tokens)?
            }

            // semicolon always required
            Expr::Binding(_) | Expr::Assign(_) => map(symbol(';'), Some)(tokens)?,

            // semicolon required if not final expression
            Expr::Infix(_)
            | Expr::Ident(_)
            | Expr::Constructor(_)
            | Expr::Literal(_)
            | Expr::Control(_)
            | Expr::Call(_) => match symbol(';')(tokens) {
                Ok((rest, semi_token)) => (rest, Some(semi_token)),
                Err(_) => {
                    sequence.push(ExprSeq {
                        expr,
                        semi_token: None,
                    });
                    break;
                }
            },
        };

        sequence.push(ExprSeq { expr, semi_token });
        tokens = rest;
    }

    let (rest, close_curly) = symbol('}')(tokens)?;

    let block = ExprBlock {
        open_curly,
        sequence,
        close_curly,
    };

    Ok((rest, block))
}

fn expr_call(input: Input) -> PResult<ExprCall> {
    map(
        tuple((
            ident,
            symbol('('),
            cut(tuple((
                context("function arguments", argument_list),
                symbol(')'),
            ))),
        )),
        |(ident, open_parens, (arguments, close_parens))| ExprCall {
            ident,
            open_parens,
            arguments,
            close_parens,
        },
    )(input)
}

fn expr_binding(input: Input) -> PResult<ExprBinding> {
    map(
        tuple((
            keyword(Keyword::Let),
            cut(tuple((
                opt(keyword(Keyword::Mut)),
                ident,
                symbol('='),
                expr_inline,
            ))),
        )),
        |(let_token, (mut_token, ident, eq_token, value))| ExprBinding {
            let_token,
            mut_token,
            ident,
            eq_token,
            value,
        },
    )(input)
}

fn expr_assign(input: Input) -> PResult<ExprAssign> {
    map(
        tuple((ident, symbol('='), cut(expr_inline))),
        |(ident, eq_token, value)| ExprAssign {
            ident,
            eq_token,
            value,
        },
    )(input)
}

fn expr_loop(input: Input) -> PResult<ExprLoop> {
    map(
        tuple((keyword(Keyword::Loop), cut(expr_block))),
        |(loop_token, body)| ExprLoop { loop_token, body },
    )(input)
}

fn expr_while(input: Input) -> PResult<ExprWhile> {
    map(
        tuple((keyword(Keyword::While), cut(pair(expr_inline, expr_block)))),
        |(while_token, (condition, body))| ExprWhile {
            while_token,
            condition,
            body,
        },
    )(input)
}

fn expr_control(input: Input) -> PResult<ExprControl> {
    alt((
        map(expr_break, ExprControl::Break),
        map(expr_continue, ExprControl::Continue),
    ))(input)
}

fn expr_break(input: Input) -> PResult<ExprBreak> {
    map(keyword(Keyword::Break), |break_token| ExprBreak {
        break_token,
    })(input)
}

fn expr_continue(input: Input) -> PResult<ExprContinue> {
    map(keyword(Keyword::Continue), |continue_token| ExprContinue {
        continue_token,
    })(input)
}

fn expr_for_loop(input: Input) -> PResult<ExprFor> {
    map(
        tuple((
            keyword(Keyword::For),
            cut(tuple((
                ident,
                keyword(Keyword::In),
                expr_inline,
                expr_block,
            ))),
        )),
        |(for_token, (ident, in_token, range, body))| ExprFor {
            for_token,
            ident,
            in_token,
            range,
            body,
        },
    )(input)
}

fn expr_constructor(input: Input) -> PResult<ExprConstructor> {
    map(
        tuple((ident, symbol('{'), initializer_list, symbol('}'))),
        |(ident, open_curly, initializers, close_curly)| ExprConstructor {
            ident,
            open_curly,
            initializers,
            close_curly,
        },
    )(input)
}

fn expr_literal(input: Input) -> PResult<ExprLiteral> {
    alt((
        map(expr_bool, ExprLiteral::Bool),
        map(expr_string, ExprLiteral::String),
        map(expr_integer, ExprLiteral::Integer),
        map(expr_float, ExprLiteral::Float),
    ))(input)
}

fn expr_bool(input: Input) -> PResult<ExprBool> {
    alt((
        map(keyword(Keyword::False), |token| ExprBool {
            token,
            value: false,
        }),
        map(keyword(Keyword::True), |token| ExprBool {
            token,
            value: true,
        }),
    ))(input)
}

fn expr_string(input: Input) -> PResult<ExprString> {
    let (rest, token) = string_literal(input)?;
    let value =
        parse_string(&token.text).map_err(|e| nom::Err::Failure(Error::single(input, e)))?;
    Ok((
        rest,
        ExprString {
            token,
            value: value.into(),
        },
    ))
}

fn expr_integer(input: Input) -> PResult<ExprInteger> {
    let (rest, token) = integer_literal(input)?;
    let value =
        parse_integer(&token.text).map_err(|e| nom::Err::Failure(Error::single(input, e)))?;
    Ok((rest, ExprInteger { token, value }))
}

fn expr_float(input: Input) -> PResult<ExprFloat> {
    let (rest, token) = float_literal(input)?;
    let value = parse_float(&token.text).map_err(|e| nom::Err::Failure(Error::single(input, e)))?;
    Ok((rest, ExprFloat { token, value }))
}

fn parse_string(text: &str) -> Result<String, ErrorKind> {
    let mut result = String::with_capacity(text.len() - 1);
    let mut chars = text.chars();

    if !matches!(chars.next(), Some('"')) {
        return Err(ErrorKind::Custom("expected a leading \""));
    }

    while let Some(ch) = chars.next() {
        match ch {
            '"' if chars.as_str().is_empty() => return Ok(result),
            '"' => {
                return Err(ErrorKind::Custom(
                    "trailing characters after terminating \"",
                ))
            }

            '\\' => match chars.next() {
                None => {
                    return Err(ErrorKind::Custom(
                        "expected escaped character, found end of string",
                    ))
                }
                Some(escape) => match escape {
                    '\\' | '"' => result.push(escape),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    _ => return Err(ErrorKind::Custom("unknown escape character")),
                },
            },

            ch => result.push(ch),
        }
    }

    Err(ErrorKind::Custom("expected a terminating \""))
}

fn parse_integer(text: &str) -> Result<Integer, ErrorKind> {
    type IntParser = fn(&str) -> Result<Integer, std::num::ParseIntError>;

    let (integer_part, parse): (&str, IntParser) =
        if let Some(suffix_start) = text.find(|ch| ch == 'u' || ch == 'i') {
            let suffix = &text[suffix_start..];
            let parse: IntParser = match suffix {
                "u8" => |text: &str| text.parse().map(Integer::U8),
                "u16" => |text: &str| text.parse().map(Integer::U16),
                "u32" => |text: &str| text.parse().map(Integer::U32),
                "u64" => |text: &str| text.parse().map(Integer::U64),
                "i8" => |text: &str| text.parse().map(Integer::I8),
                "i16" => |text: &str| text.parse().map(Integer::I16),
                "i32" => |text: &str| text.parse().map(Integer::I32),
                "i64" => |text: &str| text.parse().map(Integer::I64),
                _ => return Err(ErrorKind::InvalidIntegerSuffix),
            };

            (&text[..suffix_start], parse)
        } else {
            (text, |text| Ok(Integer::Any(text.into())))
        };

    Ok(parse(integer_part)?)
}

fn parse_float(text: &str) -> Result<Float, ErrorKind> {
    type FloatParser = fn(&str) -> Result<Float, std::num::ParseFloatError>;

    let (float_part, parse): (&str, FloatParser) =
        if let Some(suffix_start) = text.find(|ch| ch == 'u' || ch == 'i') {
            let suffix = &text[suffix_start..];
            let parse: FloatParser = match suffix {
                "f32" => |text: &str| text.parse().map(Float::F32),
                "f64" => |text: &str| text.parse().map(Float::F64),
                _ => return Err(ErrorKind::InvalidFloatSuffix),
            };

            (&text[..suffix_start], parse)
        } else {
            (text, |text| text.parse().map(Float::F32))
        };

    parse(float_part).map_err(Into::into)
}

fn parse_type(input: Input) -> PResult<Type> {
    context(
        "when parsing type",
        alt((
            map(struct_type, Type::Struct),
            map(enum_type, Type::Enum),
            map(primitive, Type::Primitive),
            map(ident, Type::Alias),
            map(pair(symbol('('), symbol(')')), |_| Type::Unit),
        )),
    )(input)
}

fn struct_type(input: Input) -> PResult<Struct> {
    map(
        tuple((
            keyword(Keyword::Struct),
            cut(tuple((symbol('{'), fields, symbol('}')))),
        )),
        |(struct_token, (open_curly, fields, close_curly))| Struct {
            struct_token,
            open_curly,
            fields,
            close_curly,
        },
    )(input)
}

fn enum_type(input: Input) -> PResult<Enum> {
    map(
        tuple((
            keyword(Keyword::Enum),
            cut(tuple((symbol('{'), variants, symbol('}')))),
        )),
        |(enum_token, (open_curly, variants, close_curly))| Enum {
            enum_token,
            open_curly,
            variants,
            close_curly,
        },
    )(input)
}

fn initializer_list(input: Input) -> PResult<Vec<Initializer>> {
    terminated(
        separated_list(cut(symbol(',')), context("initializer", initializer)),
        opt(symbol(',')),
    )(input)
}

fn initializer(input: Input) -> PResult<Initializer> {
    map(
        tuple((symbol('.'), cut(tuple((ident, symbol('='), expr_inline))))),
        |(dot_token, (ident, eq_token, value))| Initializer {
            dot_token,
            ident,
            eq_token,
            value,
        },
    )(input)
}

fn argument_list(input: Input) -> PResult<Vec<Argument>> {
    terminated(
        separated_list(symbol(','), context("when parsing argument", expr_inline)),
        opt(symbol(',')),
    )(input)
}

fn fields(input: Input) -> PResult<Fields> {
    terminated(
        separated_list(symbol(','), context("when parsing field", field)),
        opt(symbol(',')),
    )(input)
}

fn field(input: Input) -> PResult<Field> {
    map(
        tuple((ident, cut(context("when parsing type", parse_type)))),
        |(ident, ty)| Field { ident, ty },
    )(input)
}

fn variants(input: Input) -> PResult<Variants> {
    terminated(
        separated_list(symbol(','), context("when parsing field", variant)),
        opt(symbol(',')),
    )(input)
}

fn variant(input: Input) -> PResult<Variant> {
    map(
        tuple((ident, opt(pair(symbol('='), parse_type)))),
        |(ident, ty)| {
            let (eq_token, ty) = match ty {
                Some((eq_token, ty)) => (Some(eq_token), Some(ty)),
                None => (None, None),
            };

            Variant {
                ident,
                eq_token,
                ty,
            }
        },
    )(input)
}

fn generics(input: Input) -> PResult<Generics> {
    map(
        tuple((
            symbol('['),
            cut(tuple((
                terminated(separated_list(symbol(','), ident), opt(symbol(','))),
                symbol(']'),
            ))),
        )),
        |(lt_token, (idents, gt_token))| Generics {
            lt_token,
            idents,
            gt_token,
        },
    )(input)
}

fn ident(input: Input) -> PResult<Ident> {
    token_kind(TokenKind::Identifier)(input)
}

fn primitive(input: Input) -> PResult<PrimitiveType> {
    error(
        ErrorKind::ExpectedPrimitive,
        map(
            token_where(|token| matches!(token.kind, TokenKind::Keyword(Keyword::Primitive(_)))),
            |token| match token.kind {
                TokenKind::Keyword(Keyword::Primitive(primitive)) => PrimitiveType {
                    token: Some(token),
                    kind: primitive,
                },
                _ => unreachable!(),
            },
        ),
    )(input)
}

fn string_literal(input: Input) -> PResult<Token> {
    token_kind(TokenKind::Literal(Literal::String))(input)
}

fn integer_literal(input: Input) -> PResult<Token> {
    token_kind(TokenKind::Literal(Literal::Integer))(input)
}

fn float_literal(input: Input) -> PResult<Token> {
    token_kind(TokenKind::Literal(Literal::Float))(input)
}

fn symbol<'a>(symbol: impl Into<Symbol>) -> impl Fn(Input<'a>) -> PResult<Token> {
    token_kind(TokenKind::Symbol(symbol.into()))
}

fn keyword<'a>(keyword: Keyword) -> impl Fn(Input<'a>) -> PResult<Token> {
    token_kind(TokenKind::Keyword(keyword))
}

fn token_kind<'a>(kind: TokenKind) -> impl Fn(Input<'a>) -> PResult<Token> {
    map_err(
        move |err| {
            let found = match err {
                ErrorKind::UnexpectedToken(found) => Some(found),
                _ => None,
            };

            ErrorKind::ExpectedToken {
                found,
                expected: kind,
            }
        },
        token_where(move |token| token.kind == kind),
    )
}

fn token_where(f: impl Fn(&Token) -> bool) -> impl Fn(Input) -> PResult<Token> {
    move |input| {
        let (rest, token) = token(input)?;

        if f(&token) {
            Ok((rest, token))
        } else {
            Err(nom::Err::Error(Error::single(
                input,
                ErrorKind::UnexpectedToken(token.kind),
            )))
        }
    }
}

fn token(input: Input) -> PResult<Token> {
    let mut tokens = input.iter();
    let token = tokens
        .next()
        .ok_or_else(|| nom::Err::Error(Error::from_error_kind(input, NomErrorKind::Eof)))?;
    Ok((tokens.as_slice(), token.clone()))
}

fn error<'a, T>(
    kind: ErrorKind,
    f: impl Fn(Input<'a>) -> PResult<'a, T>,
) -> impl Fn(Input<'a>) -> PResult<'a, T> {
    move |input| f(input).map_err(|e| e.map(|_| Error::single(input, kind.clone())))
}

fn map_err<'a, T>(
    f: impl Fn(ErrorKind) -> ErrorKind,
    parser: impl Fn(Input<'a>) -> PResult<'a, T>,
) -> impl Fn(Input<'a>) -> PResult<'a, T> {
    move |input| {
        parser(input).map_err(|e| {
            e.map(|e| Error {
                kind: f(e.kind),
                ..e
            })
        })
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Unit => write!(f, "`()`"),
            Type::Alias(ident) => write!(f, "{}", ident),
            Type::Enum(enum_type) => write!(f, "{}", enum_type),
            Type::Struct(struct_type) => write!(f, "{}", struct_type),
            Type::Primitive(primitive) => write!(f, "{}", primitive.kind),
        }
    }
}

impl Display for Struct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fields = self
            .fields
            .iter()
            .map(|field| format!("{} {}", field.ident.text, field.ty))
            .collect::<Vec<_>>()
            .join(", ");

        write!(f, "`struct {{ {} }}`", fields)
    }
}

impl Display for Enum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variants = self
            .variants
            .iter()
            .map(|variant| match &variant.ty {
                Some(ty) => format!("{} = {}", variant.ident.text, ty),
                None => format!("{}", variant.ident.text),
            })
            .collect::<Vec<_>>()
            .join(", ");

        write!(f, "`enum {{ {} }}`", variants)
    }
}
