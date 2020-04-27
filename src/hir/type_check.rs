use super::*;

#[derive(Debug)]
struct Globals<'a> {
    types: &'a HashMap<TypeId, Type>,
    constants: HashMap<ConstId, Signature>,
}

#[derive(Debug)]
struct Scope<'a> {
    globals: &'a Globals<'a>,
    locals: HashMap<LocalId, TypeId>,
}

#[derive(Debug, Clone)]
struct Signature {
    arguments: Vec<TypeId>,
    result: TypeId,
}

impl<'a> Scope<'a> {
    fn type_of(&self, ident: Identifier) -> TypeId {
        match ident {
            Identifier::Local(local) => *self.locals.get(&local).unwrap(),
            Identifier::Global(Item::Type(ty)) => ty,
            Identifier::Global(Item::Const(constant)) => {
                todo!("get type of constant: {:?}", constant)
            }
        }
    }

    fn lookup_signature(&self, ident: Identifier) -> Option<Signature> {
        match ident {
            Identifier::Global(Item::Const(constant)) => {
                self.globals.constants.get(&constant).cloned()
            }
            Identifier::Global(Item::Type(id)) => {
                let ty = self.globals.types.get(&id)?;
                match ty {
                    Type::Alias(alias) => Some(Signature {
                        arguments: vec![*alias],
                        result: id,
                    }),
                    _ => None,
                }
            }
            Identifier::Local(_) => todo!("call local function"),
        }
    }

    fn lookup_type(&self, ty: TypeId) -> &Type {
        self.globals.types.get(&ty).unwrap()
    }
}

impl Module {
    pub(super) fn type_check(&mut self) -> Result<()> {
        let mut constants = HashMap::new();
        for (id, constant) in self.namespace.constants.iter() {
            let Constant::Function(function) = constant;

            let signature = Signature {
                arguments: function.arguments.clone(),
                result: function.return_type,
            };

            constants.insert(*id, signature);
        }

        let globals = Globals {
            types: &self.namespace.types,
            constants,
        };

        for constant in self.namespace.constants.values_mut() {
            constant.type_check(&globals)?;
        }

        Ok(())
    }
}

impl TypeId {
    fn expect(self, expected: TypeId) -> Result<()> {
        if self == expected {
            Ok(())
        } else {
            Err(Error::TypeMismatch {
                found: self,
                expected,
            })
        }
    }
}

impl Constant {
    fn type_check(&mut self, globals: &Globals) -> Result<()> {
        match self {
            Constant::Function(function) => {
                let mut scope = Scope {
                    globals,
                    locals: HashMap::new(),
                };
                function
                    .body
                    .type_check(&mut scope)?
                    .expect(function.return_type)
            }
        }
    }
}

impl Expr {
    fn type_check(&mut self, scope: &mut Scope) -> Result<TypeId> {
        let ty = match &mut self.kind {
            ExprKind::Block(block) => block.type_check(scope)?,
            ExprKind::Ident(ident) => scope.type_of(*ident),
            ExprKind::Binding(binding) => binding.type_check(scope)?,
            ExprKind::Invocation(invocation) => invocation.type_check(scope)?,
            ExprKind::Access(access) => todo!("type check item access: {:?}", access),
            ExprKind::Literal(literal) => literal.type_check(),
            ExprKind::Constructor(constructor) => constructor.type_check(scope)?,
        };

        self.ty = Some(ty);
        Ok(ty)
    }
}

impl ExprBlock {
    fn type_check(&mut self, scope: &mut Scope) -> Result<TypeId> {
        let mut previous = Namespace::TYPE_UNIT;

        for expr in &mut self.sequence {
            previous.expect(Namespace::TYPE_UNIT)?;
            previous = expr.type_check(scope)?;
        }

        Ok(previous)
    }
}

impl ExprBinding {
    fn type_check(&mut self, scope: &mut Scope) -> Result<TypeId> {
        let ty = self.value.type_check(scope)?;
        scope.locals.insert(self.local, ty);
        Ok(Namespace::TYPE_UNIT)
    }
}

impl ExprInvocation {
    fn type_check(&mut self, scope: &mut Scope) -> Result<TypeId> {
        let signature = scope
            .lookup_signature(self.target)
            .ok_or_else(|| Error::NotCallable { ident: self.target })?;

        for (argument, &ty) in self.arguments.iter_mut().zip(&signature.arguments) {
            argument.type_check(scope)?.expect(ty)?;
        }

        Ok(signature.result)
    }
}

impl ExprConstructor {
    fn type_check(&mut self, scope: &mut Scope) -> Result<TypeId> {
        let mut properties = Vec::with_capacity(self.initializers.len());
        match scope.lookup_type(self.base) {
            Type::Struct { fields } => {
                for init in &self.initializers {
                    let field = fields
                        .iter()
                        .find(|field| field.id == init.property)
                        .unwrap();
                    properties.push(field.ty);
                }
            }
            _ => unreachable!("construct non-struct type"),
        };

        for (init, field) in self.initializers.iter_mut().zip(properties) {
            init.value.type_check(scope)?.expect(field)?;
        }

        Ok(self.base)
    }
}

impl ExprLiteral {
    fn type_check(&mut self) -> TypeId {
        match self {
            ExprLiteral::String(_) => Namespace::TYPE_STR,
            ExprLiteral::Integer(ExprInteger::U8(_)) => Namespace::TYPE_U8,
            ExprLiteral::Integer(ExprInteger::U16(_)) => Namespace::TYPE_U16,
            ExprLiteral::Integer(ExprInteger::U32(_)) => Namespace::TYPE_U32,
            ExprLiteral::Integer(ExprInteger::U64(_)) => Namespace::TYPE_U64,
            ExprLiteral::Integer(ExprInteger::I8(_)) => Namespace::TYPE_I8,
            ExprLiteral::Integer(ExprInteger::I16(_)) => Namespace::TYPE_I16,
            ExprLiteral::Integer(ExprInteger::I32(_)) => Namespace::TYPE_I32,
            ExprLiteral::Integer(ExprInteger::I64(_)) => Namespace::TYPE_I64,
            ExprLiteral::Float(ExprFloat::F32(_)) => Namespace::TYPE_F32,
            ExprLiteral::Float(ExprFloat::F64(_)) => Namespace::TYPE_F64,
        }
    }
}
