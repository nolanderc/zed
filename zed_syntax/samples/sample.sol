type Age = u32;

type Person = struct {
    id u32,
    name str, 
    age Age,
}

type Gender = enum {
    Female,
    Male,
    Other,
}

type Void = ();

const main = fn () -> i32 {
    println("Hello, World");

    let age = 32u32;
    let person = Person {
        .id = 1u32,
        .name = "John Wick",
        .age = age,
    };

    for x in 0..10 {
        println(person.name);
    }

    42
}

