use super::*;

use std::cell::Cell;

#[derive(Debug)]
pub(super) struct Scope<'a> {
    next_local: Rc<Cell<LocalId>>,
    items: HashMap<Rc<str>, Identifier>,
    parent: Parent<'a>,
}

#[derive(Debug)]
enum Parent<'a> {
    Scope(&'a Scope<'a>),
    Namespace(&'a Namespace),
}

impl<'a> Scope<'a> {
    pub(super) fn new(namespace: &'a Namespace) -> Scope<'a> {
        Scope {
            next_local: Rc::new(Cell::new(LocalId(0))),
            items: HashMap::new(),
            parent: Parent::Namespace(namespace),
        }
    }

    fn subscope(&'a self) -> Scope<'a> {
        Scope {
            next_local: self.next_local.clone(),
            items: HashMap::new(),
            parent: Parent::Scope(self),
        }
    }

    fn push_local(&mut self, ident: impl Into<Rc<str>>) -> LocalId {
        let id = self.next_local.get();
        self.next_local.set(LocalId(id.0 + 1));
        self.items.insert(ident.into(), Identifier::Local(id));
        id
    }

    fn lookup(&self, name: &str) -> Option<Identifier> {
        match self.items.get(name) {
            Some(item) => Some(*item),
            None => self.parent.lookup(name),
        }
    }

    fn lookup_type(&self, name: &str) -> Result<TypeId> {
        match self.lookup(name) {
            Some(Identifier::Global(Item::Type(ty))) => Ok(ty),
            None => Err(Error::ItemNotFound { ident: name.into() }),
            Some(_) => Err(Error::ExpectedType),
        }
    }

    fn get_type(&self, ty: TypeId) -> Option<&Type> {
        self.parent.namespace().types.get(&ty)
    }
}

impl Parent<'_> {
    fn lookup(&self, name: &str) -> Option<Identifier> {
        match self {
            Parent::Scope(scope) => scope.lookup(name),
            Parent::Namespace(namespace) => namespace.lookup(name).map(Identifier::Global),
        }
    }

    fn namespace(&self) -> &Namespace {
        match self {
            Parent::Scope(scope) => scope.parent.namespace(),
            Parent::Namespace(namespace) => namespace,
        }
    }
}

impl Expr {
    pub(super) fn lower(expr: &syntax::Expr, scope: &mut Scope) -> Result<ExprKind> {
        let kind = match expr {
            syntax::Expr::Block(block) => ExprKind::Block(Self::lower_block(&block, scope)?),
            syntax::Expr::Call(call) => ExprKind::Invocation(Self::lower_call(&call, scope)?),
            syntax::Expr::Binding(binding) => {
                ExprKind::Binding(Self::lower_binding(binding, scope)?)
            }
            syntax::Expr::Literal(literal) => ExprKind::Literal(Self::lower_literal(literal)),
            syntax::Expr::Constructor(constructor) => {
                ExprKind::Constructor(Self::lower_constructor(constructor, scope)?)
            }
            syntax::Expr::Ident(ident) => ExprKind::Ident(Self::lower_ident(ident, scope)?),
            syntax::Expr::ForLoop(_) => todo!("lowering: {:?}", expr),
        };

        Ok(kind)
    }

    pub(super) fn lower_block(block: &syntax::ExprBlock, scope: &mut Scope) -> Result<ExprBlock> {
        let mut scope = scope.subscope();

        let mut sequence = Vec::with_capacity(block.sequence.len());
        for seq in &block.sequence {
            let kind = Self::lower(&seq.expr, &mut scope)?;
            let ty = if seq.semi_token.is_some() {
                Some(Namespace::TYPE_UNIT)
            } else {
                None
            };

            sequence.push(Expr { ty, kind })
        }

        Ok(ExprBlock { sequence })
    }

    fn lower_call(call: &syntax::ExprCall, scope: &mut Scope) -> Result<ExprInvocation> {
        let ident = &call.ident.text;
        let target = scope.lookup(ident).ok_or_else(|| Error::ItemNotFound {
            ident: ident.to_string(),
        })?;

        let mut arguments = Vec::with_capacity(call.arguments.len());
        for arg in &call.arguments {
            let kind = Self::lower(&arg, scope)?;
            arguments.push(Expr { ty: None, kind });
        }

        Ok(ExprInvocation { target, arguments })
    }

    fn lower_constructor(
        constructor: &syntax::ExprConstructor,
        scope: &mut Scope,
    ) -> Result<ExprConstructor> {
        let base = scope.lookup_type(&constructor.ident.text)?;

        let mut properties = Vec::with_capacity(constructor.initializers.len());
        match scope.get_type(base).unwrap() {
            Type::Struct { fields } => {
                for init in &constructor.initializers {
                    let property = fields
                        .iter()
                        .find(|field| field.name == init.ident.text)
                        .ok_or_else(|| Error::UndefinedField {
                            ident: init.ident.text.to_string(),
                        })?
                        .id;

                    properties.push(property)
                }
            }
            _ => return Err(Error::ExpectedStruct),
        };

        let mut initializers = Vec::with_capacity(constructor.initializers.len());
        for (init, property) in constructor.initializers.iter().zip(properties) {
            let value = Expr {
                ty: None,
                kind: Self::lower(&init.value, scope)?,
            };

            initializers.push(ExprInitializer {
                property,
                value: Box::new(value),
            })
        }

        Ok(ExprConstructor { base, initializers })
    }

    fn lower_binding(binding: &syntax::ExprBinding, scope: &mut Scope) -> Result<ExprBinding> {
        let value = Expr {
            ty: None,
            kind: Self::lower(&binding.value, scope)?,
        };

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

    fn lower_literal(literal: &syntax::ExprLiteral) -> ExprLiteral {
        match literal {
            syntax::ExprLiteral::String(string) => {
                ExprLiteral::String(ExprString(string.value.clone()))
            }
            syntax::ExprLiteral::Integer(integer) => ExprLiteral::Integer(match integer.value {
                syntax::Integer::U8(value) => ExprInteger::U8(value),
                syntax::Integer::U16(value) => ExprInteger::U16(value),
                syntax::Integer::U32(value) => ExprInteger::U32(value),
                syntax::Integer::U64(value) => ExprInteger::U64(value),
                syntax::Integer::I8(value) => ExprInteger::I8(value),
                syntax::Integer::I16(value) => ExprInteger::I16(value),
                syntax::Integer::I32(value) => ExprInteger::I32(value),
                syntax::Integer::I64(value) => ExprInteger::I64(value),
            }),
            syntax::ExprLiteral::Float(float) => ExprLiteral::Float(match float.value {
                syntax::Float::F32(value) => ExprFloat::F32(value),
                syntax::Float::F64(value) => ExprFloat::F64(value),
            }),
        }
    }
}
