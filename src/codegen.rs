use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::{BasicType, BasicTypeEnum, FunctionType};
use inkwell::values::FunctionValue;
use inkwell::{AddressSpace, OptimizationLevel};

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

use crate::hir;
use crate::hir::{ConstId, TypeId};
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

    if false {
        let puts_type = context.i32_type().fn_type(
            &[context.i8_type().ptr_type(AddressSpace::Generic).into()],
            true,
        );
        let puts = module.add_function("puts", puts_type, None);

        let fn_type = context.i32_type().fn_type(&[], false);
        let main = module.add_function("main", fn_type, None);

        let entry = context.append_basic_block(main, "entry");

        let builder = context.create_builder();
        builder.position_at_end(entry);

        let text = context.const_string(b"hello world", true);
        let message = module.add_global(text.get_type(), None, "text");
        message.set_initializer(&text);

        builder.build_call(
            puts,
            &[message
                .as_pointer_value()
                .const_cast(context.i8_type().ptr_type(AddressSpace::Generic))
                .into()],
            "puts_call",
        );
        let int = context.i32_type().const_int(1, false);
        builder.build_return(Some(&int));
    } else {
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
                    bodies.push((&function.body, handle));
                }
                hir::Constant::Extern(_) => {
                    let handle = module.add_function(&name, fn_type, None);
                    functions.insert(*id, handle);
                }
            }
        }

        let namespace = Namespace { types, functions };

        for (body, handle) in bodies {
            generate_expr_block(body, &namespace, handle, &module, &context);
        }
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
        Type::Unit => context.void_type().fn_type(&arguments, false),
        Type::Basic(basic) => basic.fn_type(&arguments, false),
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
    };

    Type::Basic(base)
}

fn generate_expr_block<'a>(
    block: &hir::ExprBlock,
    namespace: &Namespace,
    current: FunctionValue,
    module: &Module<'a>,
    context: &'a Context,
) {
    let entry = context.append_basic_block(current, "entry");

    let builder = context.create_builder();
    builder.position_at_end(entry);

    for seq in &block.sequence {
        match &seq.kind {
            hir::ExprKind::Invocation(invocation) => match invocation.target {
                hir::Identifier::Global(hir::Item::Const(constant)) => {
                    let function = *namespace.functions.get(&constant).unwrap();

                    let mut arguments = Vec::with_capacity(invocation.arguments.len());
                    for arg in &invocation.arguments {
                        let value = match &arg.kind {
                            hir::ExprKind::Literal(hir::ExprLiteral::String(string)) => {
                                let text = context.const_string(string.0.as_bytes(), true);
                                let message = module.add_global(text.get_type(), None, "text");
                                message.set_initializer(&text);

                                message
                                    .as_pointer_value()
                                    .const_cast(
                                        context.i8_type().ptr_type(AddressSpace::Generic).into(),
                                    )
                                    .into()
                            }

                            _ => todo!("generate_expr: {:?}", arg),
                        };

                        arguments.push(value);
                    }

                    builder.build_call(function, &arguments, "call_site");
                }
                _ => todo!("invoke local or type constructor"),
            },

            _ => todo!("generate_expr: {:?}", seq),
        }
    }

    builder.build_return(None);
}
