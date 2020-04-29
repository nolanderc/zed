use super::*;

impl<'a> Scope<'a> {
    pub fn type_of(&self, ident: Identifier) -> TypeId {
        match ident {
            Identifier::Local(local) => *self.locals.get(&local).unwrap(),
            Identifier::Global(Item::Type(ty)) => ty,
            Identifier::Global(Item::Const(constant)) => {
                todo!("get type of constant: {:?}", constant)
            }
        }
    }

}

impl TypeId {
    pub fn expect(self, expected: TypeId) -> Result<()> {
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

impl ExprKind {
    pub fn type_check(&self, scope: &mut Scope) -> Result<TypeId> {
        let ty = match &self {
            ExprKind::Block(block) => block.type_check()?,
            ExprKind::Ident(ident) => scope.type_of(*ident),
            ExprKind::Binding(binding) => binding.type_check(scope),
            ExprKind::Invocation(invocation) => invocation.type_check(scope)?,
            ExprKind::Access(access) => access.type_check(scope)?,
            ExprKind::Literal(literal) => literal.type_check(),
            ExprKind::Constructor(constructor) => constructor.type_check(scope)?,
        };

        Ok(ty)
    }
}

impl ExprBlock {
    pub fn type_check(&self) -> Result<TypeId> {
        let mut previous = Namespace::TYPE_UNIT;

        for expr in &self.sequence {
            previous.expect(Namespace::TYPE_UNIT)?;
            previous = expr.ty;
        }

        Ok(previous)
    }
}

impl ExprBinding {
    pub fn type_check(&self, scope: &mut Scope) -> TypeId {
        let ty = self.value.ty;
        scope.locals.insert(self.local, ty);
        Namespace::TYPE_UNIT
    }
}

impl ExprInvocation {
    pub fn type_check(&self, scope: &mut Scope) -> Result<TypeId> {
        let signature = scope
            .lookup_signature(self.target)
            .ok_or_else(|| Error::NotCallable { ident: self.target })?;

        for (argument, &ty) in self.arguments.iter().zip(&signature.arguments) {
            argument.ty.expect(ty)?;
        }

        Ok(signature.result)
    }
}

impl ExprConstructor {
    pub fn type_check(&self, scope: &Scope) -> Result<TypeId> {
        let mut properties = Vec::with_capacity(self.initializers.len());
        match scope.get_type(self.base) {
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

        for (init, field) in self.initializers.iter().zip(properties) {
            init.value.ty.expect(field)?;
        }

        Ok(self.base)
    }
}

impl ExprAccess {
    pub fn type_check(&self, scope: &Scope) -> Result<TypeId> {
        let ty = match scope.get_type(self.base.ty) {
            Type::Struct { fields } => fields
                .iter()
                .find(|field| field.id == self.property)
                .map(|field| field.ty)
                .unwrap(),
            Type::Enum { .. } => self.base.ty,
            Type::Primitive(primitive) => {
                return Err(Error::AccessPrimitive {
                    primitive: *primitive,
                })
            }
            Type::Unit => return Err(Error::AccessUnit),
            Type::Alias(_) => todo!("access through alias"),
        };
        Ok(ty)
    }
}

impl ExprLiteral {
    pub fn type_check(&self) -> TypeId {
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
