mod lowering;
mod type_check;

use std::collections::HashMap;
use std::convert::TryInto;
use std::rc::Rc;

use thiserror::Error;

use crate::lexer::Primitive;
use crate::syntax;

#[derive(Debug)]
pub struct Module {
    namespace: Namespace,
}

#[derive(Debug)]
pub struct Namespace {
    items: HashMap<Rc<str>, Item>,

    types: HashMap<TypeId, Type>,
    constants: HashMap<ConstId, Constant>,

    next_type: TypeId,
    next_const: ConstId,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Item {
    Type(TypeId),
    Const(ConstId),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConstId(u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LocalId(u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct PropertyId(u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Identifier {
    Global(Item),
    Local(LocalId),
}

#[derive(Debug)]
pub enum Type {
    Alias(TypeId),
    Struct { fields: Vec<Property> },
    Enum { variants: Vec<Property> },
}

#[derive(Debug)]
pub struct Property {
    name: Rc<str>,
    id: PropertyId,
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
    body: ExprBlock,
}

#[derive(Debug)]
struct Expr {
    ty: Option<TypeId>,
    kind: ExprKind,
}

#[derive(Debug)]
enum ExprKind {
    Block(ExprBlock),
    Invocation(ExprInvocation),
    Binding(ExprBinding),
    Ident(Identifier),
    Access(ExprAccess),
    Constructor(ExprConstructor),
    Literal(ExprLiteral),
}

#[derive(Debug)]
struct ExprBlock {
    sequence: Vec<Expr>,
}

#[derive(Debug)]
struct ExprInvocation {
    target: Identifier,
    arguments: Vec<Expr>,
}

#[derive(Debug)]
struct ExprBinding {
    local: LocalId,
    value: Box<Expr>,
}

#[derive(Debug)]
struct ExprAccess {
    base: Box<Expr>,
    property: PropertyId,
}

#[derive(Debug)]
struct ExprConstructor {
    base: TypeId,
    initializers: Vec<ExprInitializer>,
}

#[derive(Debug)]
struct ExprInitializer {
    property: PropertyId,
    value: Box<Expr>,
}

#[derive(Debug, Clone)]
pub enum ExprLiteral {
    String(ExprString),
    Integer(ExprInteger),
    Float(ExprFloat),
}

#[derive(Debug, Clone)]
pub struct ExprString(Rc<str>);

#[derive(Debug, Clone)]
pub enum ExprInteger {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
}

#[derive(Debug, Clone)]
pub enum ExprFloat {
    F32(f32),
    F64(f64),
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("duplicate definition of `{ident}`")]
    DuplicateIdent { ident: syntax::Ident },

    #[error("unknown reference to `{ident}`")]
    ItemNotFound { ident: String },
    #[error("expected a type")]
    ExpectedType,

    #[error("type not allowed on argument")]
    InvalidArgumentType,
    #[error("type not allowed as return argument")]
    InvalidReturnType,
    #[error("type not allowed on field")]
    InvalidFieldType,

    #[error("expected a struct")]
    ExpectedStruct,

    #[error("attempted to call a non-callable type")]
    NotCallable { ident: Identifier },

    #[error("undefined field `{ident}`")]
    UndefinedField { ident: String },

    #[error("type mismatch: expected {expected:?}, found {found:?}")]
    TypeMismatch { found: TypeId, expected: TypeId },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Namespace {
    pub fn new() -> Namespace {
        Namespace {
            items: HashMap::new(),
            next_type: TypeId(128),
            next_const: ConstId(128),

            types: HashMap::new(),
            constants: HashMap::new(),
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

    const TYPE_UNIT: TypeId = TypeId(0);
    const TYPE_U8: TypeId = TypeId(1);
    const TYPE_U16: TypeId = TypeId(2);
    const TYPE_U32: TypeId = TypeId(3);
    const TYPE_U64: TypeId = TypeId(4);
    const TYPE_I8: TypeId = TypeId(5);
    const TYPE_I16: TypeId = TypeId(6);
    const TYPE_I32: TypeId = TypeId(7);
    const TYPE_I64: TypeId = TypeId(8);
    const TYPE_F32: TypeId = TypeId(9);
    const TYPE_F64: TypeId = TypeId(10);
    const TYPE_STR: TypeId = TypeId(11);

    pub fn primitive_type(primitive: Primitive) -> TypeId {
        match primitive {
            Primitive::U8 => Self::TYPE_U8,
            Primitive::U16 => Self::TYPE_U16,
            Primitive::U32 => Self::TYPE_U32,
            Primitive::U64 => Self::TYPE_U64,
            Primitive::I8 => Self::TYPE_I8,
            Primitive::I16 => Self::TYPE_I16,
            Primitive::I32 => Self::TYPE_I32,
            Primitive::I64 => Self::TYPE_I64,
            Primitive::F32 => Self::TYPE_F32,
            Primitive::F64 => Self::TYPE_F64,
            Primitive::Str => Self::TYPE_STR,
        }
    }

    pub fn insert(&mut self, ident: syntax::Ident, item: Item) -> Result<()> {
        match self.items.insert(ident.text.clone(), item) {
            Some(_) => Err(Error::DuplicateIdent { ident }),
            None => Ok(()),
        }
    }

    pub fn lookup(&self, ident: &str) -> Option<Item> {
        self.items.get(ident).copied()
    }

    pub fn lookup_type(&self, ident: &str) -> Result<TypeId> {
        match self.lookup(ident) {
            Some(Item::Type(ty)) => Ok(ty),
            None => Err(Error::ItemNotFound {
                ident: ident.into(),
            }),
            Some(_) => Err(Error::ExpectedType),
        }
    }
}

impl Module {
    pub fn load(ast: &syntax::Ast) -> Result<Module> {
        let mut module = Module {
            namespace: Namespace::new(),
        };

        let mut namespace = Namespace::new();

        let items = module.allocate_items(&mut namespace, &ast.items)?;
        module.register_items(items)?;
        module.type_check()?;

        Ok(module)
    }

    fn register_items(&mut self, items: Vec<(&syntax::Item, Item)>) -> Result<()> {
        for (item, id) in items {
            match (item, id) {
                (syntax::Item::Type(item), Item::Type(id)) => {
                    let ty = Type::from_syntax(&item.decl.ty, &self.namespace)?;
                    self.namespace.types.insert(id, ty);
                }

                (syntax::Item::Const(item), Item::Const(id)) => {
                    let constant = Constant::from_syntax(&item.const_init, &self.namespace)?;
                    self.namespace.constants.insert(id, constant);
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
    ) -> Result<Vec<(&'a syntax::Item, Item)>> {
        let mut ids = Vec::new();

        for item in items {
            match item {
                syntax::Item::Type(ty_item) => {
                    let id = Item::Type(namespace.allocate_type());
                    self.namespace.insert(ty_item.ident.clone(), id)?;
                    ids.push((item, id));
                }

                syntax::Item::Const(const_item) => {
                    let id = Item::Const(namespace.allocate_constant());
                    self.namespace.insert(const_item.ident.clone(), id)?;
                    ids.push((item, id));
                }
            }
        }

        Ok(ids)
    }
}

impl Type {
    pub fn unit() -> Type {
        Type::Alias(Namespace::TYPE_UNIT)
    }

    pub fn from_syntax(ty: &syntax::Type, namespace: &Namespace) -> Result<Type> {
        let ty = match ty {
            syntax::Type::Unit => Type::unit(),
            syntax::Type::Primitive(primitive) => {
                Type::Alias(Namespace::primitive_type(primitive.kind))
            }
            syntax::Type::Alias(ident) => Type::Alias(namespace.lookup_type(&ident.text)?),
            syntax::Type::Struct(data) => {
                let fields = data
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, field)| {
                        let id = PropertyId(i.try_into().unwrap());
                        let name = field.ident.text.clone();
                        let ty = Type::from_syntax(&field.ty, namespace)?
                            .id()
                            .ok_or(Error::InvalidFieldType)?;
                        Ok(Property { id, name, ty })
                    })
                    .collect::<Result<_>>()?;

                Type::Struct { fields }
            }
            syntax::Type::Enum(data) => {
                let variants = data
                    .variants
                    .iter()
                    .enumerate()
                    .map(|(i, variant)| {
                        let id = PropertyId(i.try_into().unwrap());
                        let name = variant.ident.text.clone();
                        let ty = match &variant.ty {
                            None => Namespace::TYPE_UNIT,
                            Some(ty) => Type::from_syntax(ty, namespace)?
                                .id()
                                .ok_or(Error::InvalidFieldType)?,
                        };
                        Ok(Property { id, name, ty })
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
    pub fn from_syntax(init: &syntax::ConstInit, namespace: &Namespace) -> Result<Constant> {
        let constant = match init {
            syntax::ConstInit::Function(function) => {
                let arguments = function
                    .arguments
                    .iter()
                    .map(|arg| {
                        Type::from_syntax(&arg.ty, namespace)?
                            .id()
                            .ok_or(Error::InvalidArgumentType)
                    })
                    .collect::<Result<_>>()?;

                let return_type = match &function.return_clause {
                    None => Namespace::TYPE_UNIT,
                    Some(ret) => Type::from_syntax(&ret.ty, namespace)?
                        .id()
                        .ok_or(Error::InvalidReturnType)?,
                };

                let mut scope = lowering::Scope::new(namespace);
                let body = Expr::lower_block(&function.body, &mut scope)?;

                Constant::Function(Function {
                    arguments,
                    return_type,
                    body,
                })
            }
        };

        Ok(constant)
    }
}
