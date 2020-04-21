use std::collections::HashMap;
use std::rc::Rc;

use thiserror::Error;

use crate::lexer::Primitive;
use crate::syntax;

#[derive(Debug)]
pub struct Module {
    scope: Scope,
    types: HashMap<TypeId, Rc<Type>>,
    constants: HashMap<ConstId, Rc<Constant>>,
}

#[derive(Debug)]
pub struct Scope {
    items: HashMap<Rc<str>, Item>,
}

#[derive(Debug, Copy, Clone)]
pub enum Item {
    Global(GlobalItem),
}

#[derive(Debug, Copy, Clone)]
pub enum GlobalItem {
    Type(TypeId),
    Const(ConstId),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConstId(u32);

#[derive(Debug)]
pub enum Type {
    Alias(TypeId),
    Struct { fields: Vec<Field> },
    Enum { variants: Vec<Field> },
}

#[derive(Debug)]
pub struct Field {
    name: Rc<str>,
    ty: TypeId,
}

#[derive(Debug)]
pub enum Constant {
    Function(Function),
}

#[derive(Debug)]
pub struct Function {
    arguments: Vec<TypeId>,
    return_type: TypeId,
    body: Rc<Expr>,
}

#[derive(Debug)]
struct Expr;

#[derive(Debug)]
pub struct Namespace {
    next_type: TypeId,
    next_const: ConstId,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("duplicate definition of `{ident}`")]
    DuplicateIdent { ident: syntax::Ident },

    #[error("unknown reference `{path}`")]
    PathNotFound { path: syntax::Path },
    #[error("expected a type")]
    ExpectedType,

    #[error("type not allowed on argument")]
    InvalidArgumentType,
    #[error("type not allowed as return argument")]
    InvalidReturnType,
    #[error("type not allowed on field")]
    InvalidFieldType,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Namespace {
    pub fn new() -> Namespace {
        Namespace {
            next_type: TypeId(128),
            next_const: ConstId(128),
        }
    }

    pub fn allocate_type(&mut self) -> TypeId {
        let id = self.next_type;
        self.next_type.0 += 1;
        id
    }

    pub fn allocate_constant(&mut self) -> ConstId {
        let id = self.next_const;
        self.next_const.0 += 1;
        id
    }

    pub fn unit_type() -> TypeId {
        TypeId(0)
    }

    pub fn primitive_type(primitive: Primitive) -> TypeId {
        match primitive {
            Primitive::U8 => TypeId(1),
            Primitive::U16 => TypeId(2),
            Primitive::U32 => TypeId(3),
            Primitive::U64 => TypeId(4),
            Primitive::I8 => TypeId(5),
            Primitive::I16 => TypeId(6),
            Primitive::I32 => TypeId(7),
            Primitive::I64 => TypeId(8),
            Primitive::F32 => TypeId(9),
            Primitive::F64 => TypeId(10),
            Primitive::Str => TypeId(11),
        }
    }
}

impl Scope {
    pub fn new() -> Scope {
        Scope {
            items: HashMap::new(),
        }
    }

    pub fn insert(&mut self, ident: syntax::Ident, item: Item) -> Result<()> {
        match self.items.insert(ident.text.clone(), item) {
            Some(_) => Err(Error::DuplicateIdent { ident }),
            None => Ok(()),
        }
    }

    pub fn lookup(&self, path: &syntax::Path) -> Result<Item> {
        match path.idents.as_slice() {
            [name] => self
                .items
                .get(&name.text)
                .copied()
                .ok_or_else(|| Error::PathNotFound { path: path.clone() }),
            _ => Err(Error::PathNotFound { path: path.clone() }),
        }
    }

    pub fn lookup_type(&self, path: &syntax::Path) -> Result<TypeId> {
        match self.lookup(path)? {
            Item::Global(GlobalItem::Type(ty)) => Ok(ty),
            _ => Err(Error::ExpectedType),
        }
    }
}

impl Module {
    pub fn load(ast: &syntax::Ast) -> Result<Module> {
        let mut module = Module {
            scope: Scope::new(),
            types: HashMap::new(),
            constants: HashMap::new(),
        };

        let mut namespace = Namespace::new();

        let items = module.allocate_items(&mut namespace, &ast.items)?;
        module.register_items(items)?;
        module.type_check()?;

        Ok(module)
    }

    fn register_items(&mut self, items: Vec<(&syntax::Item, GlobalItem)>) -> Result<()> {
        for (item, id) in items {
            match (item, id) {
                (syntax::Item::Type(item), GlobalItem::Type(id)) => {
                    let scope = self.global_scope();
                    let ty = Type::from_syntax(&item.decl.ty, scope)?;
                    self.types.insert(id, ty.into());
                }

                (syntax::Item::Const(item), GlobalItem::Const(id)) => {
                    let scope = self.global_scope();
                    let constant = Constant::from_syntax(&item.const_init, scope)?;
                    self.constants.insert(id, constant.into());
                }

                _ => unreachable!(),
            }
        }

        Ok(())
    }

    fn allocate_items<'a>(
        &mut self,
        namespace: &mut Namespace,
        items: &'a [syntax::Item],
    ) -> Result<Vec<(&'a syntax::Item, GlobalItem)>> {
        let mut ids = Vec::new();

        for item in items {
            match item {
                syntax::Item::Type(ty_item) => {
                    let id = GlobalItem::Type(namespace.allocate_type());
                    self.scope.insert(ty_item.ident.clone(), Item::Global(id))?;
                    ids.push((item, id));
                }

                syntax::Item::Const(const_item) => {
                    let id = GlobalItem::Const(namespace.allocate_constant());
                    self.scope
                        .insert(const_item.ident.clone(), Item::Global(id))?;
                    ids.push((item, id));
                }
            }
        }

        Ok(ids)
    }

    fn global_scope(&self) -> &Scope {
        &self.scope
    }

    fn type_check(&mut self) -> Result<(), Error> {
        todo!("type checking of function bodies")
    }
}

impl Type {
    pub fn unit() -> Type {
        Type::Alias(Namespace::unit_type())
    }

    pub fn from_syntax(ty: &syntax::Type, scope: &Scope) -> Result<Type> {
        let ty = match ty {
            syntax::Type::Unit => Type::unit(),
            syntax::Type::Primitive(primitive) => {
                Type::Alias(Namespace::primitive_type(primitive.kind))
            }
            syntax::Type::Alias(path) => Type::Alias(scope.lookup_type(path)?),
            syntax::Type::Struct(data) => {
                let fields = data
                    .fields
                    .iter()
                    .map(|field| {
                        let name = field.ident.text.clone();
                        let ty = Type::from_syntax(&field.ty, scope)?
                            .id()
                            .ok_or(Error::InvalidFieldType)?;
                        Ok(Field { name, ty })
                    })
                    .collect::<Result<_>>()?;

                Type::Struct { fields }
            }
            syntax::Type::Enum(data) => {
                let variants = data
                    .variants
                    .iter()
                    .map(|variant| {
                        let name = variant.ident.text.clone();
                        let ty = match &variant.ty {
                            None => Namespace::unit_type(),
                            Some(ty) => Type::from_syntax(ty, scope)?
                                .id()
                                .ok_or(Error::InvalidFieldType)?,
                        };
                        Ok(Field { name, ty })
                    })
                    .collect::<Result<_>>()?;

                Type::Enum { variants }
            }
        };

        Ok(ty)
    }

    pub fn id(&self) -> Option<TypeId> {
        match self {
            Type::Alias(id) => Some(*id),
            Type::Struct { .. } => None,
            Type::Enum { .. } => None,
        }
    }
}

impl Constant {
    pub fn from_syntax(init: &syntax::ConstInit, scope: &Scope) -> Result<Constant> {
        let constant = match init {
            syntax::ConstInit::Function(function) => {
                let arguments = function
                    .arguments
                    .iter()
                    .map(|arg| {
                        Type::from_syntax(&arg.ty, scope)?
                            .id()
                            .ok_or(Error::InvalidArgumentType)
                    })
                    .collect::<Result<_>>()?;

                let return_type = match &function.return_clause {
                    None => Namespace::unit_type().into(),
                    Some(ret) => Type::from_syntax(&ret.ty, scope)?
                        .id()
                        .ok_or(Error::InvalidReturnType)?,
                };

                Constant::Function(Function {
                    arguments,
                    return_type,
                    body: Expr.into(),
                })
            }
        };

        Ok(constant)
    }
}
