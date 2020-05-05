#![allow(dead_code)]

mod disjoint_set;

use thiserror::Error;
use self::disjoint_set::DisjointSet;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Term(usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Variable(usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Constructor(usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TypeId(usize);

#[derive(Debug)]
pub enum Expression {
    Term(Term),
    Call(Call),
    Abstraction(Abstraction),
}

#[derive(Debug)]
pub struct Call {
    pub function: Box<Expression>,
    pub argument: Box<Expression>,
}

#[derive(Debug)]
pub struct Abstraction {
    pub parameters: Vec<Term>,
    pub expression: Box<Expression>,
}

#[derive(Debug, Clone)]
pub enum Type {
    Mono(Variable),
    Application(Application),
}

#[derive(Debug, Clone)]
pub struct Application {
    pub constructor: Constructor,
    pub arguments: Vec<TypeId>,
}

#[derive(Debug)]
pub struct ConstructorInfo {
    arity: usize,
}

#[derive(Debug)]
struct VariableInfo {
    quantified: bool,
}

#[derive(Debug)]
pub struct Context {
    /// Every term is assumed to have a specific type.
    assumptions: Vec<TypeId>,

    types: Vec<Type>,
    variables: Vec<VariableInfo>,
    construtors: Vec<ConstructorInfo>,

    /// Type equivalence classes.
    classes: DisjointSet,
}

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("type mismatch: found {found:?}, expected {expected:?}")]
    TypeMismatch {
        found: TypeId,
        expected: TypeId,
    },
}

impl Context {
    pub fn new() -> Context {
        Context {
            assumptions: Vec::new(),

            types: Vec::new(),
            variables: Vec::new(),
            construtors: vec![ConstructorInfo::mapping()],

            classes: DisjointSet::new(),
        }
    }

    pub fn add_assumption(&mut self, ty: TypeId) -> Term {
        let id = self.assumptions.len();
        self.assumptions.push(ty);
        Term(id)
    }

    pub fn infer_type(&mut self, expr: &Expression) -> Result<TypeId, Error> {
        match expr {
            Expression::Term(term) => {
                let assumed = self.assumptions[term.0];
                let ty = self.instantiate(assumed);
                Ok(ty)
            }
            Expression::Call(call) => {
                let function = self.infer_type(&call.function)?;
                let argument = self.infer_type(&call.argument)?;
                let output_ty = self.add_mono();
                let application = self.add_mapping(argument, output_ty.clone());
                self.unify(function, application)?;
                Ok(output_ty)
            }
            _ => todo!("infer type for expression"),
        }
    }

    pub fn add_mapping(&mut self, input: TypeId, output: TypeId) -> TypeId {
        let kind = Type::Application(Application {
            constructor: Constructor(0),
            arguments: vec![input, output],
        });
        self.add_type(kind)
    }

    pub fn add_variable(&mut self) -> TypeId {
        let id = self.variables.len();
        self.variables.push(VariableInfo { quantified: true });
        let variable = Variable(id);
        let kind = Type::Mono(variable);
        self.add_type(kind)
    }

    fn add_mono(&mut self) -> TypeId {
        let id = self.variables.len();
        self.variables.push(VariableInfo { quantified: false });
        let variable = Variable(id);
        let kind = Type::Mono(variable);
        self.add_type(kind)
    }

    pub fn add_type(&mut self, kind: Type) -> TypeId {
        let id = self.classes.add();
        self.types.push(kind);
        TypeId(id)
    }

    pub fn add_constructor(&mut self, info: ConstructorInfo) -> Constructor {
        let id = self.construtors.len();
        self.construtors.push(info);
        Constructor(id)
    }

    pub fn construct(&mut self, constructor: Constructor, arguments: Vec<TypeId>) -> TypeId {
        let kind = Type::Application(Application {
            constructor,
            arguments,
        });
        self.add_type(kind)
    }

    pub fn instantiate(&mut self, ty: TypeId) -> TypeId {
        match self.types[ty.0].clone() {
            Type::Mono(variable) => {
                if self.variables[variable.0].quantified {
                    self.add_mono()
                } else {
                    ty
                }
            }
            Type::Application(mut application) => {
                for arg in &mut application.arguments {
                    *arg = self.instantiate(*arg);
                }

                let kind = Type::Application(Application {
                    constructor: application.constructor,
                    arguments: application.arguments,
                });

                self.add_type(kind)
            }
        }
    }

    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<(), Error> {
        let a = self.classes.find(a.0);
        let b = self.classes.find(b.0);

        if a == b {
            return Ok(());
        }

        match (&self.types[a], &self.types[b]) {
            (Type::Application(a), Type::Application(b))
                if a.constructor == b.constructor && a.arguments.len() == b.arguments.len() =>
            {
                let mut pairs = Vec::with_capacity(a.arguments.len());

                for (a, b) in a.arguments.iter().zip(&b.arguments) {
                    pairs.push((*a, *b));
                }

                for (a, b) in pairs {
                    self.unify(a, b)?;
                }

                Ok(())
            }

            (Type::Mono(_), _) | (_, Type::Mono(_)) => {
                self.classes.union(a, b);
                Ok(())
            }

            _ => Err(Error::TypeMismatch {
                expected: TypeId(a),
                found: TypeId(b),
            }),
        }
    }

    pub fn type_eq(&mut self, a: TypeId, b: TypeId) -> bool {
        self.classes.find(a.0) == self.classes.find(b.0)
    }
}

impl ConstructorInfo {
    pub fn mapping() -> ConstructorInfo {
        ConstructorInfo { arity: 2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_call() {
        let mut context = Context::new();

        let int = context.add_constructor(ConstructorInfo { arity: 0 });
        let int = context.construct(int, vec![]);

        let float = context.add_constructor(ConstructorInfo { arity: 0 });
        let float = context.construct(float, vec![]);

        let f_ty = context.add_mapping(int, int);
        let f = context.add_assumption(f_ty);

        let x = context.add_assumption(int);

        let expr = Expression::Call(Call {
            function: Expression::Term(f).into(),
            argument: Expression::Term(x).into(),
        });

        let found = context.infer_type(&expr).unwrap();

        dbg!(int, found, &context);

        assert!(context.type_eq(int, found));
    }
}
