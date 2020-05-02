use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::{BasicType, BasicTypeEnum, FunctionType};
use inkwell::values::{
    BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue, PointerValue,
};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

use std::collections::HashMap;
use std::convert::TryInto;
use std::path::Path;

use thiserror::Error;

use crate::hir;
use crate::hir::{ConstId, LabelId, LocalId, TypeId};
use crate::lexer;

#[derive(Debug, Error)]
pub enum Error {}

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Clone)]
enum Type<'a> {
    Unit,
    Basic(BasicTypeEnum<'a>),
}

struct Namespace<'a> {
    types: HashMap<TypeId, Type<'a>>,
    functions: HashMap<ConstId, FunctionValue<'a>>,
}

pub fn emit_object(hir_module: &hir::Module, path: impl AsRef<Path>) -> Result<()> {
    let context = Context::create();

    let module = context.create_module("test_module");
    let types = create_types(&hir_module, &context);

    let namespace = &hir_module.namespace;

    let mut functions = HashMap::new();
    let mut bodies = Vec::new();

    for (id, constant) in namespace.constants.iter() {
        let signature = namespace.signatures.get(&(*id).into()).unwrap();
        let fn_type = signature_fn_type(signature, &types, &context);

        let name = namespace.names.get(&(*id).into()).unwrap();

        match constant {
            hir::Constant::Function(function) => {
                let handle = module.add_function(&name, fn_type, None);
                functions.insert(*id, handle);
                bodies.push((function, handle));
            }
            hir::Constant::Extern(_) => {
                let handle = module.add_function(&name, fn_type, None);
                functions.insert(*id, handle);
            }
        }
    }

    let namespace = Namespace { types, functions };

    for (function, handle) in bodies {
        let builder = context.create_builder();
        let entry = context.append_basic_block(handle, "entry");
        builder.position_at_end(entry);

        let mut encoder = FunctionEncoder {
            function: handle,
            namespace: &namespace,
            module: &module,
            context: &context,
            builder: &builder,
            locals: HashMap::new(),
            labels: HashMap::new(),
        };

        for (i, param) in handle.get_param_iter().enumerate() {
            let local = LocalId(i as u32);
            let pointer = encoder.get_or_insert_local(local, param.get_type());
            encoder.builder.build_store(pointer, param);
        }

        let value = encoder.generate_expr_block(&function.body);
        encoder
            .builder
            .build_return(value.as_ref().map(|v| v as &dyn BasicValue));
    }

    module.print_to_stderr();

    let init_config = InitializationConfig {
        asm_parser: true,
        asm_printer: true,
        base: true,
        disassembler: true,
        info: true,
        machine_code: true,
    };

    Target::initialize_x86(&init_config);

    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).unwrap();

    let cpu = TargetMachine::get_host_cpu_name().to_string();
    let machine = target
        .create_target_machine(
            &triple,
            &cpu,
            "",
            OptimizationLevel::None,
            RelocMode::Default,
            CodeModel::Default,
        )
        .unwrap();

    module.verify().unwrap();

    machine
        .write_to_file(&module, FileType::Object, path.as_ref())
        .unwrap();

    Ok(())
}

fn signature_fn_type<'a>(
    signature: &hir::Signature,
    types: &HashMap<TypeId, Type<'a>>,
    context: &'a Context,
) -> FunctionType<'a> {
    let return_type = types.get(&signature.result).unwrap();

    let mut arguments = Vec::with_capacity(signature.arguments.len());
    for argument in &signature.arguments {
        let arg_type = match types.get(argument).unwrap() {
            Type::Unit => todo!("unit type as function argument"),
            Type::Basic(basic) => *basic,
        };
        arguments.push(arg_type);
    }

    match return_type {
        Type::Unit => context.void_type().fn_type(&arguments, signature.variadic),
        Type::Basic(basic) => basic.fn_type(&arguments, signature.variadic),
    }
}

fn create_types<'a>(module: &hir::Module, context: &'a Context) -> HashMap<TypeId, Type<'a>> {
    let mut types = HashMap::new();

    for &id in module.namespace.types.keys() {
        create_type(id, &mut types, &module.namespace, context);
    }

    types
}

fn create_type<'a>(
    id: TypeId,
    types: &mut HashMap<TypeId, Type<'a>>,
    namespace: &hir::Namespace,
    context: &'a Context,
) {
    let definition = namespace.types.get(&id).unwrap();
    let ty = match definition {
        hir::Type::Unit => Type::Unit,
        hir::Type::Alias(id) => get_or_create_type(*id, types, namespace, context),
        hir::Type::Primitive(primitive) => primitive_type(*primitive, context),
        hir::Type::Enum { .. } => todo!("create LLVM type for enum"),
        hir::Type::Struct { fields } => {
            let mut field_types = Vec::new();
            for field in fields {
                match get_or_create_type(field.ty, types, namespace, context) {
                    Type::Basic(basic) => field_types.push(basic),
                    Type::Unit => todo!("unit type in struct"),
                }
            }

            let ty = context.struct_type(&field_types, false).into();
            Type::Basic(ty)
        }
    };

    types.insert(id, ty);
}

fn get_or_create_type<'a>(
    id: TypeId,
    types: &mut HashMap<TypeId, Type<'a>>,
    namespace: &hir::Namespace,
    context: &'a Context,
) -> Type<'a> {
    match types.get(&id) {
        Some(ty) => ty.clone(),
        None => {
            create_type(id, types, namespace, context);
            types.get(&id).unwrap().clone()
        }
    }
}

fn primitive_type(primitive: lexer::Primitive, context: &Context) -> Type {
    let base = match primitive {
        lexer::Primitive::U8 => context.i8_type().into(),
        lexer::Primitive::U16 => context.i16_type().into(),
        lexer::Primitive::U32 => context.i32_type().into(),
        lexer::Primitive::U64 => context.i64_type().into(),
        lexer::Primitive::I8 => context.i8_type().into(),
        lexer::Primitive::I16 => context.i16_type().into(),
        lexer::Primitive::I32 => context.i32_type().into(),
        lexer::Primitive::I64 => context.i64_type().into(),
        lexer::Primitive::F32 => context.f32_type().into(),
        lexer::Primitive::F64 => context.f64_type().into(),
        lexer::Primitive::Str => context.i8_type().ptr_type(AddressSpace::Generic).into(),
        lexer::Primitive::Bool => context.bool_type().into(),
    };

    Type::Basic(base)
}

struct FunctionEncoder<'ctx, 'a> {
    function: FunctionValue<'ctx>,
    namespace: &'a Namespace<'ctx>,
    module: &'a Module<'ctx>,
    context: &'ctx Context,
    builder: &'a Builder<'ctx>,
    locals: HashMap<LocalId, PointerValue<'ctx>>,
    labels: HashMap<LabelId, Label<'ctx>>,
}

struct Label<'ctx> {
    start: BasicBlock<'ctx>,
    end: BasicBlock<'ctx>,
}

impl<'ctx, 'a> FunctionEncoder<'ctx, 'a> {
    /// Get or create a new local variable
    fn get_or_insert_local(
        &mut self,
        local: LocalId,
        ty: BasicTypeEnum<'ctx>,
    ) -> PointerValue<'ctx> {
        let FunctionEncoder {
            function,
            context,
            ref mut locals,
            ..
        } = self;

        *locals.entry(local).or_insert_with(|| {
            let entry = function.get_first_basic_block().unwrap();
            let builder = context.create_builder();
            match entry.get_terminator() {
                Some(terminator) => builder.position_before(&terminator),
                None => builder.position_at_end(entry),
            }
            builder.build_alloca(ty, "local")
        })
    }

    fn generate_expr(&mut self, ast: &hir::Expr) -> Option<BasicValueEnum<'ctx>> {
        match &ast.kind {
            hir::ExprKind::Block(block) => self.generate_expr_block(block),
            hir::ExprKind::Literal(literal) => self.generate_literal(literal).into(),
            hir::ExprKind::Invocation(invocation) => self.generate_invocation(invocation),
            hir::ExprKind::Binding(binding) => {
                self.generate_binding(binding);
                None
            }
            hir::ExprKind::Assign(assign) => {
                self.generate_assign(assign);
                None
            }
            hir::ExprKind::Ident(identifier) => self.generate_identifier(*identifier),
            hir::ExprKind::Branch(branch) => self.generate_branch(branch),
            hir::ExprKind::Access(_) => todo!("generate access struct member"),
            hir::ExprKind::Constructor(_) => todo!("generate struct constructor"),
            hir::ExprKind::Loop(endless) => {
                self.generate_loop(endless);
                None
            }
            hir::ExprKind::Jump(jump) => {
                self.generate_jump(jump);
                None
            }
        }
    }

    fn generate_expr_block(&mut self, ast: &hir::ExprBlock) -> Option<BasicValueEnum<'ctx>> {
        let mut value = None;

        for seq in &ast.sequence {
            value = self.generate_expr(seq);
        }

        value
    }

    fn generate_invocation(
        &mut self,
        invocation: &hir::ExprInvocation,
    ) -> Option<BasicValueEnum<'ctx>> {
        let mut arguments = Vec::with_capacity(invocation.arguments.len());

        for arg in &invocation.arguments {
            let value = self.generate_expr(arg).unwrap();
            arguments.push(value.try_into().unwrap());
        }

        match invocation.target {
            hir::Identifier::Global(hir::Item::Const(constant)) => {
                self.generate_function_call(constant, &arguments)
            }
            _ => todo!("invoke {:?}", invocation.target),
        }
    }

    fn generate_function_call(
        &mut self,
        target: ConstId,
        arguments: &[BasicValueEnum<'ctx>],
    ) -> Option<BasicValueEnum<'ctx>> {
        let function = *self.namespace.functions.get(&target).unwrap();
        let call = self
            .builder
            .build_call(function, arguments, "function_call");
        call.try_as_basic_value().map_left(Some).left_or(None)
    }

    fn generate_literal(&mut self, literal: &hir::ExprLiteral) -> BasicValueEnum<'ctx> {
        match literal {
            hir::ExprLiteral::Bool(boolean) => self.generate_bool(boolean).into(),
            hir::ExprLiteral::String(string) => self.generate_string(string).into(),
            hir::ExprLiteral::Integer(integer) => self.generate_integer(integer).into(),
            hir::ExprLiteral::Float(float) => self.generate_float(float).into(),
        }
    }

    fn generate_bool(&mut self, boolean: &hir::ExprBool) -> IntValue<'ctx> {
        self.context.bool_type().const_int(boolean.0 as u64, false)
    }

    fn generate_string(&mut self, string: &hir::ExprString) -> PointerValue<'ctx> {
        let text = self.context.const_string(string.0.as_bytes(), true);
        let global = self.module.add_global(text.get_type(), None, "string");
        global.set_initializer(&text);
        global
            .as_pointer_value()
            .const_cast(self.context.i8_type().ptr_type(AddressSpace::Generic))
    }

    fn generate_integer(&mut self, integer: &hir::ExprInteger) -> IntValue<'ctx> {
        match integer {
            hir::ExprInteger::U8(value) => self.context.i8_type().const_int(*value as u64, false),
            hir::ExprInteger::U16(value) => self.context.i16_type().const_int(*value as u64, false),
            hir::ExprInteger::U32(value) => self.context.i32_type().const_int(*value as u64, false),
            hir::ExprInteger::U64(value) => self.context.i64_type().const_int(*value as u64, false),
            hir::ExprInteger::I8(value) => self.context.i8_type().const_int(*value as u64, true),
            hir::ExprInteger::I16(value) => self.context.i16_type().const_int(*value as u64, true),
            hir::ExprInteger::I32(value) => self.context.i32_type().const_int(*value as u64, true),
            hir::ExprInteger::I64(value) => self.context.i64_type().const_int(*value as u64, true),
        }
    }

    fn generate_float(&mut self, float: &hir::ExprFloat) -> FloatValue<'ctx> {
        match float {
            hir::ExprFloat::F32(value) => self.context.f32_type().const_float(*value as f64),
            hir::ExprFloat::F64(value) => self.context.f32_type().const_float(*value),
        }
    }

    fn generate_binding(&mut self, binding: &hir::ExprBinding) {
        let value = self.generate_expr(&binding.value);
        if let Some(value) = value {
            let local = self.get_or_insert_local(binding.local, value.get_type());
            self.builder.build_store(local, value);
        }
    }

    fn generate_assign(&mut self, assign: &hir::ExprAssign) {
        let value = self.generate_expr(&assign.value);
        if let Some(value) = value {
            let local = self.locals.get(&assign.local).unwrap();
            self.builder.build_store(*local, value);
        }
    }

    fn generate_identifier(&mut self, identifier: hir::Identifier) -> Option<BasicValueEnum<'ctx>> {
        match identifier {
            hir::Identifier::Local(local) => {
                let local = self.locals.get(&local).unwrap();
                self.builder.build_load(*local, "load_local").into()
            }
            _ => todo!("access {:?}", identifier),
        }
    }

    fn generate_branch(&mut self, branch: &hir::ExprBranch) -> Option<BasicValueEnum<'ctx>> {
        let condition = self
            .generate_expr(&branch.condition)
            .unwrap()
            .into_int_value();

        let zero = self.context.bool_type().const_zero();
        let comparison =
            self.builder
                .build_int_compare(IntPredicate::NE, condition, zero, "comparison");

        let success = self.context.append_basic_block(self.function, "then");
        let failure = self.context.append_basic_block(self.function, "else");
        let join = self.context.append_basic_block(self.function, "join");

        self.builder
            .build_conditional_branch(comparison, success, failure);

        self.builder.position_at_end(success);
        let success_value = self.generate_expr(&branch.success);
        if self.block_not_terminated() {
            self.builder.build_unconditional_branch(join);
        }

        self.builder.position_at_end(failure);
        let failure_value = self.generate_expr(&branch.failure);
        if self.block_not_terminated() {
            self.builder.build_unconditional_branch(join);
        }

        self.builder.position_at_end(join);

        match (success_value, failure_value) {
            (Some(success_value), Some(failure_value)) => {
                let value = self
                    .builder
                    .build_phi(success_value.get_type(), "branch_result");
                value.add_incoming(&[(&success_value, success), (&failure_value, failure)]);
                Some(value.as_basic_value())
            }
            (None, None) => None,
            _ => unreachable!(),
        }
    }

    fn generate_loop(&mut self, endless: &hir::ExprLoop) {
        let body = self.context.append_basic_block(self.function, "loop");
        let done = self.context.append_basic_block(self.function, "done");

        self.labels.insert(
            endless.label,
            Label {
                start: body,
                end: done,
            },
        );

        self.builder.build_unconditional_branch(body);
        self.builder.position_at_end(body);
        self.generate_expr(&endless.expr);

        if self.block_not_terminated() {
            self.builder.build_unconditional_branch(body);
        }

        self.builder.position_at_end(done);
    }

    fn generate_jump(&mut self, jump: &hir::ExprJump) {
        let label = self.labels.get(&jump.label).expect("label not in scope");
        match jump.kind {
            hir::Jump::Break => {
                self.builder.build_unconditional_branch(label.end);
            }
            hir::Jump::Continue => {
                self.builder.build_unconditional_branch(label.start);
            }
        }
    }

    fn block_not_terminated(&self) -> bool {
        self.builder
            .get_insert_block()
            .expect("builder not pointing to block")
            .get_terminator()
            .is_none()
    }
}
