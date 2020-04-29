use super::*;
use std::str::FromStr;

impl Expr {
    pub(super) fn lower(
        expr: &syntax::Expr,
        hint: Option<TypeId>,
        scope: &mut Scope,
    ) -> Result<ExprKind> {
        let kind = match expr {
            syntax::Expr::Block(block) => ExprKind::Block(Self::lower_block(&block, hint, scope)?),
            syntax::Expr::Call(call) => ExprKind::Invocation(Self::lower_call(&call, scope)?),
            syntax::Expr::Binding(binding) => {
                ExprKind::Binding(Self::lower_binding(binding, scope)?)
            }
            syntax::Expr::Literal(literal) => {
                ExprKind::Literal(Self::lower_literal(literal, hint)?)
            }
            syntax::Expr::Constructor(constructor) => {
                ExprKind::Constructor(Self::lower_constructor(constructor, scope)?)
            }
            syntax::Expr::Ident(ident) => ExprKind::Ident(Self::lower_ident(ident, scope)?),
            syntax::Expr::Infix(infix) => Self::lower_infix(infix, scope)?,
            _ => todo!("lower {:?}", expr),
        };

        Ok(kind)
    }

    pub(super) fn lower_block(
        block: &syntax::ExprBlock,
        hint: Option<TypeId>,
        scope: &mut Scope,
    ) -> Result<ExprBlock> {
        let mut scope = scope.subscope();

        let mut sequence = Vec::with_capacity(block.sequence.len());
        for (i, seq) in block.sequence.iter().enumerate() {
            let is_last = i == block.sequence.len() - 1;
            let has_semi = seq.semi_token.is_some();

            let expected = if is_last && !has_semi { hint } else { None };

            let kind = Self::lower(&seq.expr, expected, &mut scope)?;
            let ty = kind.type_check(&mut scope)?;

            let ty = if has_semi { Namespace::TYPE_UNIT } else { ty };

            sequence.push(Expr { ty, kind })
        }

        Ok(ExprBlock { sequence })
    }

    fn lower_call(call: &syntax::ExprCall, scope: &mut Scope) -> Result<ExprInvocation> {
        let ident = &call.ident.text;
        let target = scope.lookup(ident).ok_or_else(|| Error::ItemNotFound {
            ident: ident.to_string(),
        })?;

        let signature = scope
            .lookup_signature(target)
            .ok_or(Error::NotCallable { ident: target })?;

        let mut arguments = Vec::with_capacity(call.arguments.len());
        for (arg, &expected) in call.arguments.iter().zip(&signature.arguments) {
            let kind = Self::lower(&arg, Some(expected), scope)?;
            let ty = kind.type_check(scope)?;
            arguments.push(Expr { ty, kind });
        }

        Ok(ExprInvocation { target, arguments })
    }

    fn lower_constructor(
        constructor: &syntax::ExprConstructor,
        scope: &mut Scope,
    ) -> Result<ExprConstructor> {
        let base = scope.lookup_type(&constructor.ident.text)?;

        let mut properties = Vec::with_capacity(constructor.initializers.len());
        match scope.get_type(base) {
            Type::Struct { fields } => {
                for init in &constructor.initializers {
                    let property = fields
                        .iter()
                        .find(|field| field.name == init.ident.text)
                        .ok_or_else(|| Error::UndefinedField {
                            ident: init.ident.text.to_string(),
                        })?;

                    properties.push((property.id, property.ty))
                }
            }
            _ => return Err(Error::ExpectedStruct),
        };

        let mut initializers = Vec::with_capacity(constructor.initializers.len());
        for (init, (property, hint)) in constructor.initializers.iter().zip(properties) {
            let kind = Self::lower(&init.value, Some(hint), scope)?;
            let ty = kind.type_check(scope)?;

            let value = Expr { ty, kind };

            initializers.push(ExprInitializer {
                property,
                value: Box::new(value),
            })
        }

        Ok(ExprConstructor { base, initializers })
    }

    fn lower_binding(binding: &syntax::ExprBinding, scope: &mut Scope) -> Result<ExprBinding> {
        let kind = Self::lower(&binding.value, None, scope)?;
        let ty = kind.type_check(scope)?;

        let value = Expr { ty, kind };

        Ok(ExprBinding {
            local: scope.push_local(binding.ident.text.clone()),
            value: Box::new(value),
        })
    }

    fn lower_ident(ident: &syntax::Ident, scope: &Scope) -> Result<Identifier> {
        match scope.lookup(&ident.text) {
            Some(ident) => Ok(ident),
            None => Err(Error::ItemNotFound {
                ident: ident.text.to_string(),
            }),
        }
    }

    fn lower_literal(
        literal: &syntax::ExprLiteral,
        expected: Option<TypeId>,
    ) -> Result<ExprLiteral> {
        let literal = match literal {
            syntax::ExprLiteral::String(string) => {
                ExprLiteral::String(ExprString(string.value.clone()))
            }
            syntax::ExprLiteral::Integer(integer) => ExprLiteral::Integer(match &integer.value {
                syntax::Integer::U8(value) => ExprInteger::U8(*value),
                syntax::Integer::U16(value) => ExprInteger::U16(*value),
                syntax::Integer::U32(value) => ExprInteger::U32(*value),
                syntax::Integer::U64(value) => ExprInteger::U64(*value),
                syntax::Integer::I8(value) => ExprInteger::I8(*value),
                syntax::Integer::I16(value) => ExprInteger::I16(*value),
                syntax::Integer::I32(value) => ExprInteger::I32(*value),
                syntax::Integer::I64(value) => ExprInteger::I64(*value),
                syntax::Integer::Any(value) => {
                    fn parse_integer<T: FromStr>(text: &str) -> Result<T> {
                        text.parse().map_err(|_| Error::OutOfRange)
                    }
                    match expected {
                        None => ExprInteger::I32(parse_integer(&value)?),
                        Some(Namespace::TYPE_U8) => ExprInteger::U8(parse_integer(&value)?),
                        Some(Namespace::TYPE_U16) => ExprInteger::U16(parse_integer(&value)?),
                        Some(Namespace::TYPE_U32) => ExprInteger::U32(parse_integer(&value)?),
                        Some(Namespace::TYPE_U64) => ExprInteger::U64(parse_integer(&value)?),
                        Some(Namespace::TYPE_I8) => ExprInteger::I8(parse_integer(&value)?),
                        Some(Namespace::TYPE_I16) => ExprInteger::I16(parse_integer(&value)?),
                        Some(Namespace::TYPE_I32) => ExprInteger::I32(parse_integer(&value)?),
                        Some(Namespace::TYPE_I64) => ExprInteger::I64(parse_integer(&value)?),
                        Some(expected) => {
                            return Err(Error::TypeMismatch {
                                expected,
                                found: Namespace::TYPE_I32,
                            })
                        }
                    }
                }
            }),
            syntax::ExprLiteral::Float(float) => ExprLiteral::Float(match float.value {
                syntax::Float::F32(value) => ExprFloat::F32(value),
                syntax::Float::F64(value) => ExprFloat::F64(value),
            }),
        };
        Ok(literal)
    }

    fn lower_infix(infix: &syntax::Infix, scope: &mut Scope) -> Result<ExprKind> {
        let kind = match infix.operator {
            syntax::BinaryOperator::Dot => Self::lower_operator_dot(&infix.lhs, &infix.rhs, scope)?,
            _ => todo!("lower operator: {:?}", infix.operator),
        };

        Ok(kind)
    }

    fn lower_operator_dot(
        lhs: &syntax::Expr,
        rhs: &syntax::Expr,
        scope: &mut Scope,
    ) -> Result<ExprKind> {
        let base = Self::lower(lhs, None, scope)?;
        let base_ty = base.type_check(scope)?;
        let ty = scope.get_type(base_ty);

        match ty {
            Type::Struct { fields } => match rhs {
                syntax::Expr::Ident(ident) => {
                    let property = fields
                        .iter()
                        .find(|field| field.name == ident.text)
                        .map(|field| field.id)
                        .ok_or_else(|| Error::UndefinedField {
                            ident: ident.text.to_string(),
                        })?;

                    Ok(ExprKind::Access(ExprAccess {
                        base: Box::new(Expr {
                            ty: base_ty,
                            kind: base,
                        }),
                        property,
                    }))
                }
                _ => Err(Error::ExpectedIdentifier),
            },
            Type::Enum { .. } => match rhs {
                syntax::Expr::Ident(_) => todo!("construct enum variant"),
                _ => Err(Error::ExpectedIdentifier),
            },
            Type::Primitive(primitive) => Err(Error::AccessPrimitive {
                primitive: *primitive,
            }),
            Type::Unit => Err(Error::AccessUnit),
            _ => Err(Error::ExpectedStruct),
        }
    }
}
