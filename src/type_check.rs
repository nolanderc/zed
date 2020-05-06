#![allow(dead_code)]

mod disjoint_set;

use self::disjoint_set::DisjointSet;
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Term(usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Constructor(usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(usize);

#[derive(Debug)]
pub enum Expression {
    Term(Term),
    Call(Call),
}

#[derive(Debug)]
pub struct Call {
    pub function: Box<Expression>,
    pub argument: Box<Expression>,
}

#[derive(Debug)]
pub enum Type {
    Unkonwn,
    Composite(Constructor, Vec<Type>),
}

#[derive(Debug, Clone)]
struct Application {
    pub constructor: Constructor,
    pub arguments: Vec<TypeId>,
}

#[derive(Debug)]
pub struct Context {
    /// Terms can be assumed to have a specific type.
    assumptions: Vec<TypeId>,

    types: Vec<TypeData>,
    constructors: Vec<ConstructorInfo>,
    constraints: Vec<Application>,

    /// TypeShape equivalence classes.
    classes: DisjointSet,

    next_variable: usize,
}

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("type mismatch: found {found:?}, expected {expected:?}")]
    TypeShapeMismatch { found: TypeId, expected: TypeId },

    #[error("type mismatch: found {found:?}, expected {expected:?}")]
    GenericArgumentCount { found: usize, expected: usize },

    #[error("type did not implement {constraint:?}")]
    ConstraintNotMet { constraint: Constructor },
}

#[derive(Debug)]
struct ConstructorInfo {
    arity: usize,
    impls: Vec<Constraint>,
}

#[derive(Debug, Clone)]
struct TypeData {
    shape: TypeShape,
}

#[derive(Debug, Clone)]
enum TypeShape {
    Variable(Variable),
    Application(Application),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Constraint(usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Variable {
    binding: Binding,
    restrictions: Vec<Constraint>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Binding {
    Bound,
    Free,
}

impl TypeData {
    pub fn from_shape(shape: TypeShape) -> TypeData {
        TypeData { shape }
    }
}

impl Variable {
    pub fn new(binding: Binding) -> Variable {
        Variable {
            binding,
            restrictions: Vec::new(),
        }
    }
}

impl Context {
    pub fn new() -> Context {
        Context {
            assumptions: Vec::new(),
            types: Vec::new(),
            constructors: vec![ConstructorInfo::mapping()],
            constraints: Vec::new(),

            classes: DisjointSet::new(),

            next_variable: 0,
        }
    }

    #[instrument(skip(self))]
    pub fn assumption(&mut self, ty: TypeId) -> Term {
        let id = self.assumptions.len();
        self.assumptions.push(ty);
        let term = Term(id);
        debug!(?term, ?ty, "assume");
        term
    }

    #[instrument(skip(self))]
    pub fn mapping(&mut self, input: TypeId, output: TypeId) -> TypeId {
        let shape = TypeShape::Application(Application {
            constructor: Constructor(0),
            arguments: vec![input, output],
        });

        let data = TypeData::from_shape(shape);
        self.add_type(data)
    }

    #[instrument(skip(self))]
    pub fn generic(&mut self) -> TypeId {
        let shape = TypeShape::Variable(Variable::new(Binding::Bound));
        let data = TypeData::from_shape(shape);
        self.add_type(data)
    }

    #[instrument(skip(self))]
    pub fn placeholder(&mut self) -> TypeId {
        let shape = TypeShape::Variable(Variable::new(Binding::Free));
        let data = TypeData::from_shape(shape);
        self.add_type(data)
    }

    #[instrument(skip(self))]
    fn add_type(&mut self, data: TypeData) -> TypeId {
        let id = self.classes.add();
        self.types.push(data);
        let ty = TypeId(id);
        debug!(?ty, "added type");
        ty
    }

    #[instrument(skip(self))]
    pub fn constructor(&mut self, arity: usize) -> Constructor {
        let id = self.constructors.len();
        let info = ConstructorInfo {
            arity,
            impls: Vec::new(),
        };
        self.constructors.push(info);
        Constructor(id)
    }

    #[instrument(skip(self))]
    pub fn construct(&mut self, constructor: Constructor, arguments: Vec<TypeId>) -> TypeId {
        self.try_construct(constructor, arguments).unwrap()
    }

    #[instrument(skip(self))]
    pub fn try_construct(
        &mut self,
        constructor: Constructor,
        arguments: Vec<TypeId>,
    ) -> Result<TypeId, Error> {
        let info = &self.constructors[constructor.0];
        if info.arity != arguments.len() {
            return Err(Error::GenericArgumentCount {
                found: arguments.len(),
                expected: info.arity,
            });
        }

        let shape = TypeShape::Application(Application {
            constructor,
            arguments,
        });

        let data = TypeData::from_shape(shape);
        Ok(self.add_type(data))
    }

    pub fn constraint(&mut self, constructor: Constructor, arguments: Vec<TypeId>) -> Constraint {
        let id = self.constraints.len();
        let data = Application {
            constructor,
            arguments,
        };
        self.constraints.push(data);
        Constraint(id)
    }

    /// Make the concrete type fulfill a constraint.
    #[instrument(skip(self))]
    pub fn grant(&mut self, constructor: Constructor, constraint: Constraint) {
        self.constructors[constructor.0].impls.push(constraint);
    }

    /// Ensure that the given type fulfills a constraint
    #[instrument(skip(self))]
    pub fn restrict(&mut self, ty: TypeId, constraint: Constraint) {
        match &mut self.types[ty.0].shape {
            TypeShape::Application(app) => {
                todo!("check restriction for concrete type");
            }
            TypeShape::Variable(variable) => {
                variable.restrictions.push(constraint);
            }
        }
    }

    #[instrument(skip(self))]
    pub fn infer_type(&mut self, expr: &Expression) -> Result<TypeId, Error> {
        let ty = match expr {
            Expression::Term(term) => {
                let assumed = self.assumptions[term.0];
                trace!(ty = ?assumed, "assumed");
                self.instantiate(assumed, &mut HashMap::new())
                    .unwrap_or(assumed)
            }
            Expression::Call(call) => {
                let function = self.infer_type(&call.function)?;
                let argument = self.infer_type(&call.argument)?;
                let output_ty = self.placeholder();
                let application = self.mapping(argument, output_ty);
                self.unify(function, application)?;
                output_ty
            }
        };

        debug!(?ty, "inferred");

        Ok(ty)
    }

    #[instrument(skip(self))]
    pub fn instantiate(
        &mut self,
        ty: TypeId,
        substitutions: &mut HashMap<TypeId, TypeId>,
    ) -> Option<TypeId> {
        if let Some(new) = substitutions.get(&ty) {
            debug!(?new, "already instantiated");
            return Some(*new);
        }

        let kind = &self.types[ty.0];
        trace!(?kind, "instantiating");

        let new = match &kind.shape {
            TypeShape::Variable(variable) => match variable.binding {
                Binding::Bound => {
                    let restrictions = variable.restrictions.clone();
                    let ty = self.placeholder();
                    for restriction in restrictions {
                        let mut constraint = self.constraints[restriction.0].clone();

                        let mut changed = false;
                        for arg in &mut constraint.arguments {
                            if let Some(new) = self.instantiate(*arg, substitutions) {
                                *arg = new;
                                changed = true;
                            }
                        }

                        let restriction = if changed {
                            self.constraint(constraint.constructor, constraint.arguments)
                        } else {
                            restriction
                        };

                        self.restrict(ty, restriction);
                    }
                    ty
                }
                Binding::Free => {
                    debug!("free variable");
                    return None;
                }
            },
            TypeShape::Application(application) => {
                let mut application = application.clone();
                let mut changed = false;

                for arg in &mut application.arguments {
                    if let Some(new) = self.instantiate(*arg, substitutions) {
                        *arg = new;
                        changed = true;
                    }
                }

                if !changed {
                    debug!("not further specialized");
                    return None;
                }

                let shape = TypeShape::Application(Application {
                    constructor: application.constructor,
                    arguments: application.arguments,
                });

                let data = TypeData::from_shape(shape);
                self.add_type(data)
            }
        };

        substitutions.insert(ty, new);
        debug!(?new, "instantiated");
        Some(new)
    }

    #[instrument(skip(self))]
    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<(), Error> {
        let root_a = self.classes.find(a.0);
        let root_b = self.classes.find(b.0);

        if root_a == root_b {
            trace!(?root_a, ?root_b, "found common root");
            return Ok(());
        }

        let (type_a, type_b) = get_two_mut(&mut self.types, root_a, root_b);
        trace!(?type_a, ?type_b, "unifying types");
        match (&mut type_a.shape, &mut type_b.shape) {
            (TypeShape::Application(app_a), TypeShape::Application(app_b))
                if app_a.constructor == app_b.constructor
                    && app_a.arguments.len() == app_b.arguments.len() =>
            {
                let mut pairs = Vec::with_capacity(app_a.arguments.len());

                for (arg_a, arg_b) in app_a.arguments.iter().zip(&app_b.arguments) {
                    pairs.push((*arg_a, *arg_b));
                }

                for (arg_a, arg_b) in pairs {
                    self.unify(arg_a, arg_b)?;
                }
            }

            (TypeShape::Variable(var_a), TypeShape::Variable(var_b)) => {
                if var_a.restrictions.len() >= var_b.restrictions.len() {
                    var_a.restrictions.append(&mut var_b.restrictions);
                    *var_b = var_a.clone();
                } else {
                    var_b.restrictions.append(&mut var_a.restrictions);
                    *var_a = var_b.clone();
                }
            }

            (TypeShape::Application(app), TypeShape::Variable(var)) => {
                let impls = self.constructors[app.constructor.0].impls.clone();
                let restrictions = var.restrictions.clone();
                self.unify_restrictions(restrictions, impls)?;
                self.types[root_b] = self.types[root_a].clone();
            }
            (TypeShape::Variable(var), TypeShape::Application(app)) => {
                let impls = self.constructors[app.constructor.0].impls.clone();
                let restrictions = var.restrictions.clone();
                self.unify_restrictions(restrictions, impls)?;
                self.types[root_a] = self.types[root_b].clone();
            }

            _ => {
                return Err(Error::TypeShapeMismatch {
                    expected: a,
                    found: b,
                })
            }
        }

        debug!(?root_a, ?root_b, "union");
        self.classes.union(root_a, root_b);

        Ok(())
    }

    #[instrument(skip(self))]
    fn unify_restrictions(
        &mut self,
        restrictions: Vec<Constraint>,
        impls: Vec<Constraint>,
    ) -> Result<(), Error> {
        let mut impl_arguments = HashMap::with_capacity(impls.len());
        for i in impls {
            let app = self.constraints[i.0].clone();
            impl_arguments.insert(app.constructor, app.arguments);
        }

        let mut pairs = Vec::with_capacity(restrictions.len());
        for restriction in restrictions {
            let restrict = &self.constraints[restriction.0];

            let args =
                impl_arguments
                    .get(&restrict.constructor)
                    .ok_or(Error::ConstraintNotMet {
                        constraint: restrict.constructor,
                    })?;

            if restrict.arguments.len() != args.len() {
                return Err(Error::GenericArgumentCount {
                    found: args.len(),
                    expected: restrict.arguments.len(),
                });
            }

            pairs.push((restrict.clone(), args.clone()));
        }

        for (r, i) in pairs {
            for (a, b) in r.arguments.into_iter().zip(i) {
                self.unify(a, b)?;
            }
        }

        Ok(())
    }

    pub fn type_eq(&self, a: TypeId, b: TypeId) -> bool {
        self.classes.find_immutable(a.0) == self.classes.find_immutable(b.0)
    }

    fn get_type(&self, ty: TypeId) -> Type {
        let root = self.classes.find_immutable(ty.0);
        match &self.types[root].shape {
            TypeShape::Variable(_) => Type::Unkonwn,
            TypeShape::Application(app) => {
                let mut args = Vec::with_capacity(app.arguments.len());
                for arg in app.arguments.iter() {
                    args.push(self.get_type(*arg))
                }

                Type::Composite(app.constructor, args)
            }
        }
    }
}

fn get_two_mut<T>(slice: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    if a == b {
        panic!("attemted to borrow element twice: {} == {}", a, b);
    }

    let larger = a.max(b);
    let smaller = a.min(b);

    let (first, second) = slice.split_at_mut(larger);
    let mut elem_a = &mut first[smaller];
    let mut elem_b = &mut second[0];

    if a > b {
        std::mem::swap(&mut elem_a, &mut elem_b);
    }

    (elem_a, elem_b)
}

impl ConstructorInfo {
    pub fn mapping() -> ConstructorInfo {
        ConstructorInfo {
            arity: 2,
            impls: Vec::new(),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Unkonwn => write!(f, "{{unknown}}"),
            Type::Composite(constructor, args) => {
                if constructor.0 == 0 {
                    assert_eq!(args.len(), 2);
                    write!(f, "({}) -> ({})", args[0], args[1])
                } else {
                    write!(f, "T{}", constructor.0)?;

                    for arg in args.iter() {
                        write!(f, " ({})", arg)?;
                    }

                    Ok(())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_call() {
        let mut context = Context::new();

        let float = context.constructor(0);
        let float = context.construct(float, vec![]);

        let alpha = context.generic();
        let f_ty = context.mapping(alpha, alpha);
        let f = context.assumption(f_ty);

        let x = context.assumption(float);

        let expr = Expression::Call(Call {
            function: Expression::Term(f).into(),
            argument: Expression::Term(x).into(),
        });

        let found = context.infer_type(&expr).unwrap();

        assert!(context.type_eq(found, float));
        assert!(!context.type_eq(found, alpha));
    }

    #[test]
    fn optional_type() {
        let mut context = Context::new();

        let option = context.constructor(1);
        let alpha = context.generic();
        let option_ty = context.construct(option, vec![alpha]);

        let float = context.constructor(0);
        let float_ty = context.construct(float, vec![]);

        let some_ty = context.mapping(alpha, option_ty);
        let some = context.assumption(some_ty);
        let none_ty = option_ty;
        let none = context.assumption(none_ty);

        let unwrap_ty = context.mapping(option_ty, alpha);
        let unwrap = context.assumption(unwrap_ty);

        let x_ty = float_ty;
        let x = context.assumption(x_ty);

        let some_x = Expression::Call(Call {
            function: Expression::Term(some).into(),
            argument: Expression::Term(x).into(),
        });

        let expr = Expression::Call(Call {
            function: Expression::Term(unwrap).into(),
            argument: some_x.into(),
        });

        let found = context.infer_type(&expr).unwrap();

        assert!(context.type_eq(found, float_ty));
    }

    #[test]
    fn let_bindings() {
        let mut context = Context::new();

        let int = context.constructor(0);
        let float = context.constructor(0);

        let source = context.construct(int, vec![]);
        let target = context.construct(float, vec![]);
        let int_to_float_ty = context.mapping(source, target);
        let int_to_float = context.assumption(int_to_float_ty);

        let int_constant_ty = context.construct(int, vec![]);
        let int_constant = context.assumption(int_constant_ty);
        let value = Expression::Term(int_constant);

        let binding_ty = context.infer_type(&value).unwrap();
        let var_x = context.assumption(binding_ty);

        let conversion = Expression::Call(Call {
            function: Expression::Term(int_to_float).into(),
            argument: Expression::Term(var_x).into(),
        });

        let conversion_ty = context.infer_type(&conversion).unwrap();

        assert!(context.type_eq(conversion_ty, target));
        assert!(context.type_eq(binding_ty, source));
    }

    #[test]
    fn infer_function() {
        let mut context = Context::new();

        let int = context.constructor(0);
        let float = context.constructor(0);

        // `?`
        let func_ty = context.placeholder();
        let func = context.assumption(func_ty);

        // `int`
        let arg_ty = context.construct(int, vec![]);
        let arg = context.assumption(arg_ty);

        let call = Expression::Call(Call {
            function: Expression::Term(func).into(),
            argument: Expression::Term(arg).into(),
        });

        // `?`
        let output_ty = context.infer_type(&call).unwrap();
        let output = context.assumption(output_ty);

        // `float -> float`
        let target = context.construct(float, vec![]);
        let target = context.mapping(target, target);
        let coerce_ty = context.mapping(target, target);
        let coerce = context.assumption(coerce_ty);

        let coercion = Expression::Call(Call {
            function: Expression::Term(coerce).into(),
            argument: Expression::Term(output).into(),
        });

        // force output to coerce into `float`
        let actual_ty = context.infer_type(&coercion).unwrap();

        let schema = context.get_type(func_ty);
        println!("{}", schema);
    }

    #[test]
    fn variable_constraint() {
        let mut context = Context::new();

        // type int
        let int = context.constructor(0);
        let int_ty = context.construct(int, vec![]);

        // type float
        let float = context.constructor(0);
        let float_ty = context.construct(float, vec![]);

        // trait Integral
        let integral = context.constructor(0);

        // impl Integral for int
        let int_integral = context.constraint(integral, vec![]);
        context.grant(int, int_integral);

        // T: Integral
        let index_ty = context.placeholder();
        let index_integral = context.constraint(integral, vec![]);
        context.restrict(index_ty, index_integral);

        // fn access<T: Integral>(T) -> float
        let access_ty = context.mapping(index_ty, float_ty);
        let access = context.assumption(access_ty);

        // let int_const: int;
        let int_constant = context.assumption(int_ty);

        // let float_const: float;
        let float_constant = context.assumption(float_ty);

        // access(float_const)
        let float_index = Expression::Call(Call {
            function: Expression::Term(access).into(),
            argument: Expression::Term(float_constant).into(),
        });

        macro_rules! print_ty {
            ($ident:expr) => {
                println!(
                    concat!(stringify!($ident), " ({:?}) = {}"),
                    $ident,
                    context.get_type($ident)
                );
            };
        }

        print_ty!(int_ty);
        print_ty!(float_ty);
        print_ty!(index_ty);
        print_ty!(access_ty);

        println!("--------");

        // should fail
        let result = context.infer_type(&float_index);

        print_ty!(int_ty);
        print_ty!(float_ty);
        print_ty!(index_ty);
        print_ty!(access_ty);

        println!("--------");

        dbg!(&context.classes);

        match result {
            Ok(ty) => panic!("found type: {}", context.get_type(ty)),
            Err(_) => {}
        };

        // access(int_const)
        let int_index = Expression::Call(Call {
            function: Expression::Term(access).into(),
            argument: Expression::Term(int_constant).into(),
        });

        // should work
        context.infer_type(&int_index).unwrap();
    }

    #[test]
    fn associated_trait_return_value() {
        use tracing_subscriber::layer::SubscriberExt;
        let subscriber = tracing_subscriber::Registry::default().with(crate::trace::Tree::new(2));
        tracing::subscriber::with_default(subscriber, || {
            let mut context = Context::new();

            // type int32
            let int32 = context.constructor(0);
            let int32_ty = context.construct(int32, vec![]);

            // type int64
            let int64 = context.constructor(0);
            let int64_ty = context.construct(int64, vec![]);

            // type float32
            let float32 = context.constructor(0);
            let float32_ty = context.construct(float32, vec![]);

            // type float64
            let float64 = context.constructor(0);
            let float64_ty = context.construct(float64, vec![]);

            // trait Promote<Target>
            let promote = context.constructor(1);

            // impl Promote<int64> for int32
            let int32_promote = context.constraint(promote, vec![int64_ty]);
            context.grant(int32, int32_promote);

            // impl Promote<float64> for float32
            let float32_promote = context.constraint(promote, vec![float64_ty]);
            context.grant(float32, float32_promote);

            // fn promote<T, U>(T) -> U where T: Promote<U>
            let generic_t = context.generic();
            let generic_u = context.generic();
            let generic_t_constraint = context.constraint(promote, vec![generic_u]);
            context.restrict(generic_t, generic_t_constraint);
            let promote_fn_ty = context.mapping(generic_t, generic_u);
            let promote_fn = context.assumption(promote_fn_ty);

            macro_rules! print_ty {
                ($ident:expr) => {
                    println!(
                        concat!(stringify!($ident), " ({:?}) = {}"),
                        $ident,
                        context.get_type($ident)
                    );
                };
            }

            print_ty!(int32_ty);
            print_ty!(int64_ty);
            print_ty!(generic_t);
            print_ty!(generic_u);
            print_ty!(promote_fn_ty);

            // let int32_const: int32;
            let int32_const = context.assumption(int32_ty);

            // promote(int32_const)
            let call_promote_int32 = Expression::Call(Call {
                function: Expression::Term(promote_fn).into(),
                argument: Expression::Term(int32_const).into(),
            });

            let int32_result = context.infer_type(&call_promote_int32).unwrap();
            println!("-----------");
            print_ty!(int32_result);

            print_ty!(int32_ty);
            print_ty!(int64_ty);
            print_ty!(generic_t);
            print_ty!(generic_u);
            print_ty!(promote_fn_ty);

            // let int64_const: int64;
            let int64_const = context.assumption(int64_ty);

            // promote(int64_const)
            let call_promote_int64 = Expression::Call(Call {
                function: Expression::Term(promote_fn).into(),
                argument: Expression::Term(int64_const).into(),
            });

            let int64_result = context.infer_type(&call_promote_int64).unwrap_err();
            println!("-----------");
            println!("{}", int64_result);

            print_ty!(int32_ty);
            print_ty!(int64_ty);
            print_ty!(generic_t);
            print_ty!(generic_u);
            print_ty!(promote_fn_ty);

            // let float32_const: float32;
            let float32_const = context.assumption(float32_ty);

            // promote(float32_const)
            let call_promote_float32 = Expression::Call(Call {
                function: Expression::Term(promote_fn).into(),
                argument: Expression::Term(float32_const).into(),
            });

            let float32_result = context.infer_type(&call_promote_float32).unwrap();
            println!("-----------");
            print_ty!(float32_result);

            print_ty!(int32_ty);
            print_ty!(int64_ty);
            print_ty!(generic_t);
            print_ty!(generic_u);
            print_ty!(promote_fn_ty);

            panic!();
        });
    }
}
