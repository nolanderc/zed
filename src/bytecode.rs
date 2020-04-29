#[derive(Debug)]
pub struct Program {
    entry: FunctionId,
    functions: Vec<Function>,
}

#[derive(Debug)]
struct FunctionId(u32);

#[derive(Debug)]
struct Function {
    blocks: Vec<Block>,
    /// number of local variables, includes local arguments.
    locals: usize,
}

/// Index of 
#[derive(Debug)]
struct LocalId(u16);

#[derive(Debug)]
struct BlockId(u16);

#[derive(Debug)]
struct Block {
    instructions: Vec<Instruction>,
}

#[derive(Debug)]
enum Instruction {
    Call(Call),
    Jump(Jump),
    Return(Return),
    Push(Push),
    Pop(Pop),
    Move(Move),
}

/// Call a different function.
#[derive(Debug)]
struct Call(FunctionId);

/// Jump to a block within the current function.
#[derive(Debug)]
struct Jump {
    block: BlockId,
}

/// Return from the current block.
#[derive(Debug)]
struct Return;

/// Push the value in a register to the stack.
#[derive(Debug)]
struct Push {
    register: Register,
}

/// Pop a value to the stack, storing the value in the regsiter.
#[derive(Debug)]
struct Pop {
    register: Register,
}

/// Copy the value in the source register to the target register.
#[derive(Debug)]
struct Move {
    source: LocalId,
    target: LocalId,
}

