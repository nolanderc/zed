use super::*;

use std::iter;
use std::str::FromStr;

impl Expr {
    fn infer_type(kind: ExprKind, scope: &mut Scope) -> Result<Expr> {
        let ty = kind.type_check(scope)?;
        Ok(Expr {
            ty,
            kind: Box::new(kind),
        })
    }

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
            syntax::Expr::Loop(endless) => ExprKind::Loop(Self::lower_loop(endless, hint, scope)?),
            syntax::Expr::While(until) => Self::lower_while(until, scope)?,
            syntax::Expr::ForLoop(for_loop) => todo!("lower for loop"),
            syntax::Expr::Control(control) => Self::lower_control(control, scope)?,
            syntax::Expr::If(branch) => ExprKind::Branch(Self::lower_if(branch, hint, scope)?),
            syntax::Expr::Assign(assign) => ExprKind::Assign(Self::lower_assign(assign, scope)?),
        };

        Ok(kind)
    }

    pub(super) fn lower_block(
        block: &syntax::ExprBlock,
        hint: Option<TypeId>,
        scope: &mut Scope,
    ) -> Result<ExprBlock> {
        let size = scope.items.len();

        let mut sequence = Vec::with_capacity(block.sequence.len());
        for (i, seq) in block.sequence.iter().enumerate() {
            let is_last = i == block.sequence.len() - 1;
            let has_semi = seq.semi_token.is_some();

            let expected = if is_last && !has_semi { hint } else { None };

            let kind = Self::lower(&seq.expr, expected, scope)?;
            let ty = kind.type_check(scope)?;

            let ty = if has_semi { Namespace::TYPE_UNIT } else { ty };

            let is_jump = matches!(kind, ExprKind::Jump(_));

            sequence.push(Expr {
                ty,
                kind: Box::new(kind),
            });

            if is_jump {
                if !is_last {
                    eprintln!("warning: unreachable expressions");
                }

                break;
            }
        }

        scope.items.truncate(size);

        Ok(ExprBlock { sequence })
    }

    fn lower_if(
        branch: &syntax::ExprIf,
        hint: Option<TypeId>,
        scope: &mut Scope,
    ) -> Result<ExprBranch> {
        let condition = Self::lower(&branch.condition, Some(Namespace::TYPE_BOOL), scope)?;
        let condition = Self::infer_type(condition, scope)?;

        let success = Self::lower_block(&branch.block, hint, scope)?;
        let success = Self::infer_type(ExprKind::Block(success), scope)?;

        let failure = match &branch.else_branch {
            None => {
                let block = ExprBlock {
                    sequence: Vec::new(),
                };
                Self::infer_type(ExprKind::Block(block), scope)?
            }
            Some(otherwise) => {
                let failure = Self::lower(&otherwise.expr, hint, scope)?;
                Self::infer_type(failure, scope)?
            }
        };

        Ok(ExprBranch {
            condition,
            success,
            failure,
        })
    }

    fn lower_call(call: &syntax::ExprCall, scope: &mut Scope) -> Result<ExprInvocation> {
        let ident = &call.ident.text;
        let target = scope.lookup(ident).ok_or_else(|| Error::ItemNotFound {
            ident: ident.to_string(),
        })?;

        let signature = scope
            .lookup_signature(target)
            .ok_or(Error::NotCallable { ident: target })?;

        if !signature.sufficient_arguments(call.arguments.len()) {
            return Err(Error::ArgumentCount {
                expected: signature.arguments.len(),
                found: call.arguments.len(),
            });
        }

        let mut arguments = Vec::with_capacity(call.arguments.len());
        let arg_types = signature
            .arguments
            .iter()
            .copied()
            .map(Some)
            .chain(iter::repeat(None));

        for (arg, expected) in call.arguments.iter().zip(arg_types) {
            let kind = Self::lower(&arg, expected, scope)?;
            let parameter = Self::infer_type(kind, scope)?;
            arguments.push(parameter);
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
            let value = Self::infer_type(kind, scope)?;

            initializers.push(ExprInitializer { property, value })
        }

        Ok(ExprConstructor { base, initializers })
    }

    fn lower_binding(binding: &syntax::ExprBinding, scope: &mut Scope) -> Result<ExprBinding> {
        let kind = Self::lower(&binding.value, None, scope)?;
        let value = Self::infer_type(kind, scope)?;

        Ok(ExprBinding {
            mutable: binding.mut_token.is_some(),
            local: scope.allocate_local(binding.ident.text.clone()),
            value,
        })
    }

    fn lower_assign(assign: &syntax::ExprAssign, scope: &mut Scope) -> Result<ExprAssign> {
        let kind = Self::lower(&assign.value, None, scope)?;
        let value = Self::infer_type(kind, scope)?;

        Ok(ExprAssign {
            local: scope.lookup_local(&assign.ident.text)?,
            value,
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
            syntax::ExprLiteral::Bool(boolean) => ExprLiteral::Bool(ExprBool(boolean.value)),
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

    fn lower_infix(infix: &syntax::ExprInfix, scope: &mut Scope) -> Result<ExprKind> {
        use syntax::BinaryOperator as BinOp;

        let lower_comparison =
            |kind, scope| Self::lower_operator_compare(kind, &infix.lhs, &infix.rhs, scope);

        let lower_binary_intrinsic = |intrinsic: fn(Expr, Expr) -> Intrinsic, scope: &mut Scope| {
            let lhs = Self::infer_type(Self::lower(&infix.lhs, None, scope)?, scope)?;
            let rhs = Self::infer_type(Self::lower(&infix.rhs, Some(lhs.ty), scope)?, scope)?;
            Ok(ExprKind::Intrinsic(intrinsic(lhs, rhs)))
        };

        let lower_binary_intrinsic_numeracy =
            |intrinsic: fn(Numeracy, Expr, Expr) -> Intrinsic, scope: &mut Scope| {
                let lhs = Self::infer_type(Self::lower(&infix.lhs, None, scope)?, scope)?;
                let rhs = Self::infer_type(Self::lower(&infix.rhs, Some(lhs.ty), scope)?, scope)?;
                let numeracy = lhs.ty.numeracy().unwrap();
                Ok(ExprKind::Intrinsic(intrinsic(numeracy, lhs, rhs)))
            };

        let kind = match infix.operator {
            BinOp::Dot => Self::lower_operator_dot(&infix.lhs, &infix.rhs, scope)?,

            BinOp::Add => lower_binary_intrinsic(Intrinsic::Add, scope)?,
            BinOp::Sub => lower_binary_intrinsic(Intrinsic::Sub, scope)?,
            BinOp::Mul => lower_binary_intrinsic(Intrinsic::Mul, scope)?,
            BinOp::Div => lower_binary_intrinsic_numeracy(Intrinsic::Div, scope)?,
            BinOp::Mod => lower_binary_intrinsic_numeracy(Intrinsic::Mod, scope)?,

            BinOp::Equal => lower_comparison(Comparison::Equal, scope)?,
            BinOp::NotEqual => lower_comparison(Comparison::NotEqual, scope)?,
            BinOp::LessThan => lower_comparison(Comparison::LessThan, scope)?,
            BinOp::GreaterThan => lower_comparison(Comparison::GreaterThan, scope)?,
            BinOp::LessThanEqual => lower_comparison(Comparison::LessThanEqual, scope)?,
            BinOp::GreaterThanEqual => lower_comparison(Comparison::GreaterThanEqual, scope)?,
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
                        base: Expr {
                            ty: base_ty,
                            kind: Box::new(base),
                        },
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

    fn lower_operator_compare(
        kind: Comparison,
        lhs: &syntax::Expr,
        rhs: &syntax::Expr,
        scope: &mut Scope,
    ) -> Result<ExprKind> {
        let lhs = Self::infer_type(Self::lower(lhs, None, scope)?, scope)?;
        let rhs = Self::infer_type(Self::lower(rhs, Some(lhs.ty), scope)?, scope)?;

        let intrinsic = if lhs.ty.is_integer_signed() {
            Intrinsic::Compare(Numeracy::Integer(Sign::Signed), kind, lhs, rhs)
        } else if lhs.ty.is_integer_unsigned() {
            Intrinsic::Compare(Numeracy::Integer(Sign::Unsigned), kind, lhs, rhs)
        } else if lhs.ty.is_float() {
            Intrinsic::Compare(Numeracy::Float, kind, lhs, rhs)
        } else {
            return Err(Error::NumericRequired);
        };

        Ok(ExprKind::Intrinsic(intrinsic))
    }

    fn lower_loop(
        endless: &syntax::ExprLoop,
        _hint: Option<TypeId>,
        scope: &mut Scope,
    ) -> Result<ExprLoop> {
        let label = scope.push_label();
        let body = Self::lower_block(&endless.body, None, scope)?;
        scope.pop_label();

        let expr = Self::infer_type(ExprKind::Block(body), scope)?;
        Ok(ExprLoop { label, expr })
    }

    fn lower_control(control: &syntax::ExprControl, scope: &mut Scope) -> Result<ExprKind> {
        let jump = |kind| {
            let label = scope.closest_label().ok_or(Error::NoEnclosingLabel)?;
            Ok(ExprKind::Jump(ExprJump { label, kind }))
        };

        match control {
            syntax::ExprControl::Break(_) => jump(Jump::Break),
            syntax::ExprControl::Continue(_) => jump(Jump::Continue),
            syntax::ExprControl::Return(ret) => {
                let value = match &ret.value {
                    Some(value) => {
                        let value = Self::lower(value, Some(scope.signature.result), scope)?;
                        let value = Self::infer_type(value, scope)?;
                        Some(value)
                    }
                    None => None,
                };
                Ok(ExprKind::Return(ExprReturn { value }))
            }
        }
    }

    fn lower_while(until: &syntax::ExprWhile, scope: &mut Scope) -> Result<ExprKind> {
        let condition = Self::lower(&until.condition, Some(Namespace::TYPE_BOOL), scope)?;
        let condition = Self::infer_type(condition, scope)?;

        let label = scope.push_label();
        let body = Self::lower_block(&until.body, None, scope)?;
        let body = Self::infer_type(ExprKind::Block(body), scope)?;
        scope.pop_label();

        let branch = ExprBranch {
            condition,
            success: body,
            failure: Expr {
                ty: Namespace::TYPE_UNIT,
                kind: Box::new(ExprKind::Jump(ExprJump {
                    label,
                    kind: Jump::Break,
                })),
            },
        };
        let expr = Self::infer_type(ExprKind::Branch(branch), scope)?;

        Ok(ExprKind::Loop(ExprLoop { label, expr }))
    }
}
