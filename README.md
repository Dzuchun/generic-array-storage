```ignore
This crate does no require nor std nor alloc. You're welcome ❤️
```

Intertop between [`generic_array`] and [`nalgebra`], for `const usize`-hidden array storages.

This crate will presumably become obsolete, once const generics are introduced, but until then - feel free to unitize it.

**NOTE**: [`nalgebra`]'s storage traits are fundamentally unsafe, so **there is unsafe code inside**, lints mentioned in `Cargo.toml` are there just for extra self-control.

# Adding to a project

```toml
# Cargo.toml
[dependencies]
generic-array-storage = { git = "/* git ref here */" }
```

# Contribution

Just open an issue/PR or something. I'm happy to discuss any additions/fixes!

# Context

## What is `typenum`?

[`typenum`] implements integer operations as types. Basically, it allows for const arithmetic through same sort of trait wizardry, or something :idk:.

The takeaway is:

- if type `A` represents integer `x`
- and type `B` represents integer `y`

then

- `<A typenum::marker_traits::Unsigned>::{U8, U16, .., I8, I16, ..}` are associated constants equal to `x` (if possible)
- (same for `B` and `y`)
- `typenum::operator_aliases::Sum<A, B>` represents integer `x + y`
- `typenum::operator_aliases::Prod<A, B>` represents integer `x * y`

etc

## What is `genetic_array`?

[`generic_array`] implements arrays sized via `ArrayLength` trait implementors. Namely, it is implemented for `typenum` types, allowing creating an array sized as sum of two other arrays:

```rust,ignore
// some normal rust arrrays
let arr1: [i32; 3] = [1, 2, 3];
let arr2: [i32; 2] = [3, 5];

// some less-normal `generic_array` arrays
// (but having the same size and still allocated on stack)
let garr1 = GenericArray::from_array(arr1);
let garr2 = GenericArray::from_array(arr2);

// array concatenation
let garr_concat = GenericArray::concat(garr1, garr2);

// back to normal rust arrays
let concat: [i32; 5] = garr_concat.into_array();
// let concat: [i32; 6] = garr_concat.into_array(); // <-- does not compile!
```

Coolest thing is - this code is panic-free, and fully statically checked, and missized arrays will result in compilation error.

## What is `nalgebra`?

[`nalgebra`] is matrix manipulation library, abstracted over type actually storing matrix elements. This allows for efficient matrix storing, if their dimension(s) are statically known.

Generally, to store matrix on a stack entirely, you'll need for both of it's dimensions to be known, like `nalgrabra::U2` or `nalgebra::U3`. Unfortunately, default storage provided by `nalgebra` has a `const usize` type parameters, so they can't be used in case of sizes provided by associated constants.

## What is `generic_array_storage`?

This crate provides implementation of traits defining `nalgebra` storage backed up by `generic_array` arrays. This allows creation of matrices having dimensions backed up by `typenum`.

For ease of use, there's a `GenericMatrix` type alias, and `GenericMatrixExt` extension trait, providing convenient type interface and conversion functions respectively. Note that `GenericMatrix` is an alias to `nalgebra::Matrix`, so all of the functions provided by `nalgebra` are generally supported.

[`nalgebra`]: https://docs.rs/nalgebra/latest/nalgebra
[`generic_array`]: https://docs.rs/generic-array/latest/generic_array
[`typenum`]: https://docs.rs/typenum/latest/typenum
