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
    pub namespace: Namespace,
}

#[derive(Debug)]
pub struct Namespace {
    pub items: HashMap<Rc<str>, Item>,
    pub names: HashMap<Item, Rc<str>>,

    pub types: HashMap<TypeId, Type>,
    pub constants: HashMap<ConstId, Constant>,
    pub signatures: HashMap<Identifier, Signature>,

    pub next_type: TypeId,
    pub next_const: ConstId,
}

#[derive(Debug)]
struct Allocation<'a> {
    types: Vec<(TypeId, &'a syntax::Type)>,
    functions: Vec<(ConstId, &'a syntax::Function)>,
    externs: Vec<(ConstId, &'a syntax::ExternFunction)>,
}

#[derive(Debug)]
struct Scope<'a> {
    namespace: &'a Namespace,
    signature: &'a Signature,

    parent: Option<&'a Scope<'a>>,

    items: Vec<(Rc<str>, Identifier)>,

    next_local: LocalId,
    locals: HashMap<LocalId, Local>,

    next_label: LabelId,
    labels: Vec<LabelId>,
}

#[derive(Debug)]
pub struct Local {
    ty: TypeId,
    mutable: bool,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub arguments: Vec<TypeId>,
    pub result: TypeId,
    pub variadic: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Item {
    Type(TypeId),
    Const(ConstId),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConstId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PropertyId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LabelId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Identifier {
    Global(Item),
    Local(LocalId),
}

#[derive(Debug)]
pub enum Type {
    Unit,
    Alias(TypeId),
    Primitive(Primitive),
    Struct { fields: Vec<Property> },
    Enum { variants: Vec<Property> },
}

#[derive(Debug)]
pub struct Property {
    pub name: Rc<str>,
    pub id: PropertyId,
    pub ty: TypeId,
}

#[derive(Debug)]
pub enum Constant {
    Function(Function),
    Extern(ExternFunction),
}

#[derive(Debug)]
pub struct Function {
    pub body: ExprBlock,
    pub locals: HashMap<LocalId, Local>,
}

#[derive(Debug)]
pub struct ExternFunction;

#[derive(Debug)]
pub struct Expr {
    pub ty: TypeId,
    pub kind: Box<ExprKind>,
}

#[derive(Debug)]
pub enum ExprKind {
    Block(ExprBlock),
    Invocation(ExprInvocation),
    Binding(ExprBinding),
    Assign(ExprAssign),
    Ident(Identifier),
    Access(ExprAccess),
    Constructor(ExprConstructor),
    Literal(ExprLiteral),
    Branch(ExprBranch),
    Loop(ExprLoop),
    Jump(ExprJump),
    Return(ExprReturn),
    Intrinsic(Intrinsic),
}

#[derive(Debug)]
pub struct ExprLoop {
    pub label: LabelId,
    pub expr: Expr,
}

#[derive(Debug)]
pub struct ExprReturn {
    pub value: Option<Expr>,
}


#[derive(Debug)]
pub struct ExprJump {
    pub label: LabelId,
    pub kind: Jump,
}

#[derive(Debug)]
pub enum Jump {
    Break,
    Continue,
}

#[derive(Debug)]
pub struct ExprBlock {
    pub sequence: Vec<Expr>,
}

#[derive(Debug)]
pub struct ExprBranch {
    pub condition: Expr,
    pub success: Expr,
    pub failure: Expr,
}

#[derive(Debug)]
pub struct ExprInvocation {
    pub target: Identifier,
    pub arguments: Vec<Expr>,
}

#[derive(Debug)]
pub enum Intrinsic {
    Add(Expr, Expr),
    Sub(Expr, Expr),
    Mul(Expr, Expr),
    Div(Numeracy, Expr, Expr),
    Mod(Numeracy, Expr, Expr),

    Compare(Numeracy, Comparison, Expr, Expr),
}

#[derive(Debug, Copy, Clone)]
pub enum Numeracy {
    Integer(Sign),
    Float,
}

#[derive(Debug, Copy, Clone)]
pub enum Sign {
    Signed,
    Unsigned,
}

#[derive(Debug, Copy, Clone)]
pub enum Comparison {
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
}

#[derive(Debug)]
pub struct ExprBinding {
    pub mutable: bool,
    pub local: LocalId,
    pub value: Expr,
}

#[derive(Debug)]
pub struct ExprAssign {
    pub local: LocalId,
    pub value: Expr,
}

#[derive(Debug)]
pub struct ExprAccess {
    pub base: Expr,
    pub property: PropertyId,
}

#[derive(Debug)]
pub struct ExprConstructor {
    pub base: TypeId,
    pub initializers: Vec<ExprInitializer>,
}

#[derive(Debug)]
pub struct ExprInitializer {
    pub property: PropertyId,
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
pub struct ExprBool(pub bool);

#[derive(Debug, Clone)]
pub struct ExprString(pub Rc<str>);

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
    #[error("expected a locally bound variable")]
    ExpectedLocal,

    #[error("type not allowed on argument")]
    InvalidArgumentType,
    #[error("type not allowed as return argument")]
    InvalidReturnType,
    #[error("type not allowed on field")]
    InvalidFieldType,

    #[error("expected a struct")]
    ExpectedStruct,
    #[error("expected an identifier")]
    ExpectedIdentifier,

    #[error("cannot access through a primitive {primitive}")]
    AccessPrimitive { primitive: Primitive },
    #[error("cannot access through the unit type `()`")]
    AccessUnit,

    #[error("attempted to call a non-callable type {ident:?}")]
    NotCallable { ident: Identifier },
    #[error("attempted to assign to a non-mutable local variable")]
    NotMutable { local: LocalId },

    #[error("variadic arguments only allowed on `extern` functions")]
    DisallowedVariadic,

    #[error("undefined field `{ident}`")]
    UndefinedField { ident: String },

    #[error("type mismatch: expected {expected:?}, found {found:?}")]
    TypeMismatch { found: TypeId, expected: TypeId },

    #[error("integer out of range")]
    OutOfRange,

    #[error("argument count mismatch: expected {expected}, found {found}")]
    ArgumentCount { expected: usize, found: usize },

    #[error("no enclosing label in scope")]
    NoEnclosingLabel,

    #[error("intrinsic only allowed on numeric types")]
    NumericRequired,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Namespace {
    pub fn new() -> Namespace {
        Namespace {
            items: HashMap::new(),
            names: HashMap::new(),

            next_type: TypeId(128),
            next_const: ConstId(128),

            types: Self::built_in_types(),
            constants: HashMap::new(),
            signatures: HashMap::new(),
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

    fn built_in_types() -> HashMap<TypeId, Type> {
        let mut types = HashMap::new();

        types.insert(Self::TYPE_UNIT, Type::Unit);
        types.insert(Self::TYPE_U8, Type::Primitive(Primitive::U8));
        types.insert(Self::TYPE_U16, Type::Primitive(Primitive::U16));
        types.insert(Self::TYPE_U32, Type::Primitive(Primitive::U32));
        types.insert(Self::TYPE_U64, Type::Primitive(Primitive::U64));
        types.insert(Self::TYPE_I8, Type::Primitive(Primitive::I8));
        types.insert(Self::TYPE_I16, Type::Primitive(Primitive::I16));
        types.insert(Self::TYPE_I32, Type::Primitive(Primitive::I32));
        types.insert(Self::TYPE_I64, Type::Primitive(Primitive::I64));
        types.insert(Self::TYPE_F32, Type::Primitive(Primitive::F32));
        types.insert(Self::TYPE_F64, Type::Primitive(Primitive::F64));
        types.insert(Self::TYPE_STR, Type::Primitive(Primitive::Str));
        types.insert(Self::TYPE_BOOL, Type::Primitive(Primitive::Bool));

        types
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
    const TYPE_BOOL: TypeId = TypeId(12);

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
            Primitive::Bool => Self::TYPE_BOOL,
        }
    }

    pub fn insert(&mut self, ident: syntax::Ident, item: Item) -> Result<()> {
        match self.items.insert(ident.text.clone(), item) {
            Some(_) => Err(Error::DuplicateIdent { ident }),
            None => match self.names.insert(item, ident.text.clone()) {
                Some(_) => Err(Error::DuplicateIdent { ident }),
                None => Ok(()),
            },
        }
    }

    pub fn insert_constant(&mut self, id: ConstId, constant: impl Into<Constant>) {
        self.constants.insert(id, constant.into());
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

impl<'a> Scope<'a> {
    fn new(namespace: &'a Namespace, signature: &'a Signature) -> Scope<'a> {
        Scope {
            namespace,
            signature,
            parent: None,
            items: Vec::new(),
            next_local: LocalId(0),
            locals: HashMap::new(),
            next_label: LabelId(0),
            labels: Vec::new(),
        }
    }

    fn allocate_local(&mut self, ident: impl Into<Rc<str>>) -> LocalId {
        let id = self.next_local;
        self.next_local.0 += 1;
        self.items.push((ident.into(), Identifier::Local(id)));
        id
    }

    fn push_label(&mut self) -> LabelId {
        let id = self.next_label;
        self.next_label.0 += 1;
        self.labels.push(id);
        id
    }

    fn pop_label(&mut self) {
        self.labels.pop().expect("label stack empty");
    }

    fn closest_label(&self) -> Option<LabelId> {
        self.labels.last().copied()
    }

    fn lookup(&self, name: &str) -> Option<Identifier> {
        match self
            .items
            .iter()
            .rev()
            .find(|(ident, _)| ident.as_ref() == name)
        {
            Some((_, item)) => Some(*item),
            None => match self.parent {
                Some(parent) => parent.lookup(name),
                None => self.namespace.lookup(name).map(Identifier::Global),
            },
        }
    }

    fn lookup_type(&self, name: &str) -> Result<TypeId> {
        match self.lookup(name) {
            Some(Identifier::Global(Item::Type(ty))) => Ok(ty),
            None => Err(Error::ItemNotFound { ident: name.into() }),
            Some(_) => Err(Error::ExpectedType),
        }
    }

    pub fn lookup_local(&self, name: &str) -> Result<LocalId> {
        match self.lookup(name) {
            Some(Identifier::Local(local)) => Ok(local),
            None => Err(Error::ItemNotFound { ident: name.into() }),
            Some(_) => Err(Error::ExpectedLocal),
        }
    }

    fn get_type(&self, ty: TypeId) -> &Type {
        self.namespace.types.get(&ty).unwrap()
    }

    fn lookup_signature(&self, ident: Identifier) -> Option<&'a Signature> {
        self.namespace.signatures.get(&ident)
    }
}

impl Module {
    pub fn load(ast: &syntax::Ast) -> Result<Module> {
        let mut module = Module {
            namespace: Namespace::new(),
        };

        let mut namespace = Namespace::new();

        let allocation = module.allocate_items(&mut namespace, &ast.items)?;
        module.register_items(allocation)?;

        Ok(module)
    }

    fn allocate_items<'a>(
        &mut self,
        namespace: &mut Namespace,
        items: &'a [syntax::Item],
    ) -> Result<Allocation<'a>> {
        let mut allocation = Allocation {
            types: Vec::new(),
            functions: Vec::new(),
            externs: Vec::new(),
        };

        for item in items {
            match item {
                syntax::Item::Type(ty_item) => {
                    let id = namespace.allocate_type();
                    self.namespace
                        .insert(ty_item.ident.clone(), Item::Type(id))?;
                    allocation.types.push((id, &ty_item.decl.ty));
                }

                syntax::Item::Const(const_item) => {
                    let id = namespace.allocate_constant();
                    self.namespace
                        .insert(const_item.ident.clone(), Item::Const(id))?;
                    match &const_item.const_init {
                        syntax::ConstInit::Function(function) => {
                            allocation.functions.push((id, &function))
                        }
                        syntax::ConstInit::Extern(function) => {
                            allocation.externs.push((id, &function))
                        }
                    }
                }
            }
        }

        Ok(allocation)
    }

    fn register_items(&mut self, allocation: Allocation) -> Result<()> {
        self.register_types(&allocation)?;
        self.register_function_signatures(&allocation)?;
        self.register_functions(&allocation)?;
        Ok(())
    }

    fn register_types(&mut self, allocation: &Allocation) -> Result<()> {
        for (id, ty) in &allocation.types {
            let ty = Type::from_syntax(ty, &self.namespace)?;
            self.namespace.types.insert(*id, ty);
        }

        for (id, _) in &allocation.types {
            if let Some(signature) = Signature::from_type(*id, &self.namespace) {
                self.namespace.signatures.insert((*id).into(), signature);
            }
        }

        Ok(())
    }

    fn register_function_signatures(&mut self, allocation: &Allocation) -> Result<()> {
        let function_signatures = allocation.functions.iter().map(|(i, f)| (*i, &f.signature));
        let extern_signatures = allocation.externs.iter().map(|(i, f)| (*i, &f.signature));

        for (id, signature) in function_signatures.chain(extern_signatures) {
            let signature = Signature::from_function_signature(&signature, &self.namespace)?;
            self.namespace.signatures.insert(id.into(), signature);
        }

        Ok(())
    }

    fn register_functions(&mut self, allocation: &Allocation) -> Result<()> {
        for (id, _) in &allocation.externs {
            self.namespace.insert_constant(*id, ExternFunction);
        }

        for (id, function) in &allocation.functions {
            let ident = Identifier::Global(Item::Const(*id));
            let signature = self.namespace.signatures.get(&ident).unwrap();
            if signature.variadic {
                return Err(Error::DisallowedVariadic);
            }
            let function = Function::from_syntax(function, signature, &self.namespace)?;
            self.namespace.insert_constant(*id, function);
        }

        Ok(())
    }
}

impl TypeId {
    fn numeracy(self) -> Option<Numeracy> {
        match self {
            Namespace::TYPE_I8
            | Namespace::TYPE_I16
            | Namespace::TYPE_I32
            | Namespace::TYPE_I64 => Some(Numeracy::Integer(Sign::Signed)),
            Namespace::TYPE_U8
            | Namespace::TYPE_U16
            | Namespace::TYPE_U32
            | Namespace::TYPE_U64 => Some(Numeracy::Integer(Sign::Unsigned)),
            Namespace::TYPE_F32 | Namespace::TYPE_F64 => Some(Numeracy::Float),
            _ => None,
        }
    }

    fn is_integer_signed(self) -> bool {
        match self {
            Namespace::TYPE_I8 => true,
            Namespace::TYPE_I16 => true,
            Namespace::TYPE_I32 => true,
            Namespace::TYPE_I64 => true,
            _ => false,
        }
    }

    fn is_integer_unsigned(self) -> bool {
        match self {
            Namespace::TYPE_U8 => true,
            Namespace::TYPE_U16 => true,
            Namespace::TYPE_U32 => true,
            Namespace::TYPE_U64 => true,
            _ => false,
        }
    }

    fn is_integer(self) -> bool {
        self.is_integer_signed() || self.is_integer_unsigned()
    }

    fn is_float(self) -> bool {
        match self {
            Namespace::TYPE_F32 => true,
            Namespace::TYPE_F64 => true,
            _ => false,
        }
    }

    fn is_numeric(self) -> bool {
        self.is_integer() || self.is_float()
    }
}

impl Type {
    pub fn unit() -> Type {
        Type::Alias(Namespace::TYPE_UNIT)
    }

    pub fn from_syntax(ty: &syntax::Type, namespace: &Namespace) -> Result<Type> {
        let ty = match ty {
            syntax::Type::Unit => Type::unit(),
            syntax::Type::Primitive(primitive) => Type::Primitive(primitive.kind),
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
            Type::Unit => Some(Namespace::TYPE_UNIT),
            Type::Alias(id) => Some(*id),
            Type::Primitive(primitive) => Some(Namespace::primitive_type(*primitive)),
            Type::Struct { .. } => None,
            Type::Enum { .. } => None,
        }
    }
}

impl Function {
    fn from_syntax(
        function: &syntax::Function,
        signature: &Signature,
        namespace: &Namespace,
    ) -> Result<Function> {
        let mut scope = Scope::new(namespace, signature);

        for (arg, ty) in function
            .signature
            .arguments
            .iter()
            .zip(&signature.arguments)
        {
            let local = scope.allocate_local(arg.ident.text.clone());
            scope.locals.insert(
                local,
                Local {
                    ty: *ty,
                    mutable: false,
                },
            );
        }

        let body = Expr::lower_block(&function.body, Some(signature.result), &mut scope)?;
        body.type_check()?.expect(signature.result)?;

        Ok(Function {
            body,
            locals: scope.locals,
        })
    }
}

impl Signature {
    fn from_function_signature(
        signature: &syntax::Signature,
        namespace: &Namespace,
    ) -> Result<Signature> {
        let arguments = signature
            .arguments
            .iter()
            .map(|arg| {
                Type::from_syntax(&arg.ty, namespace)?
                    .id()
                    .ok_or(Error::InvalidArgumentType)
            })
            .collect::<Result<_>>()?;

        let result = match &signature.return_clause {
            None => Namespace::TYPE_UNIT,
            Some(ret) => Type::from_syntax(&ret.ty, namespace)?
                .id()
                .ok_or(Error::InvalidReturnType)?,
        };

        let variadic = signature.ellipses.is_some();

        Ok(Signature {
            arguments,
            result,
            variadic,
        })
    }

    fn from_type(ty: TypeId, namespace: &Namespace) -> Option<Signature> {
        match namespace.types.get(&ty).unwrap() {
            Type::Unit => Some(Signature {
                arguments: vec![Namespace::TYPE_UNIT],
                result: ty,
                variadic: false,
            }),
            Type::Alias(alias) => Some(Signature {
                arguments: vec![*alias],
                result: ty,
                variadic: false,
            }),
            Type::Primitive(primitive) => Some(Signature {
                arguments: vec![Namespace::primitive_type(*primitive)],
                result: ty,
                variadic: false,
            }),
            Type::Struct { .. } => None,
            Type::Enum { .. } => None,
        }
    }

    fn sufficient_arguments(&self, count: usize) -> bool {
        if self.variadic {
            self.arguments.len() <= count
        } else {
            self.arguments.len() == count
        }
    }
}

impl From<ConstId> for Item {
    fn from(id: ConstId) -> Self {
        Item::Const(id)
    }
}

impl From<TypeId> for Item {
    fn from(id: TypeId) -> Self {
        Item::Type(id)
    }
}

impl From<Item> for Identifier {
    fn from(id: Item) -> Self {
        Identifier::Global(id)
    }
}

impl From<ConstId> for Identifier {
    fn from(id: ConstId) -> Self {
        Item::from(id).into()
    }
}

impl From<TypeId> for Identifier {
    fn from(id: TypeId) -> Self {
        Item::from(id).into()
    }
}

impl From<Function> for Constant {
    fn from(function: Function) -> Self {
        Constant::Function(function)
    }
}

impl From<ExternFunction> for Constant {
    fn from(function: ExternFunction) -> Self {
        Constant::Extern(function)
    }
}
