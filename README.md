![CI status](https://github.com/Dzuchun/generic_array_storage/actions/workflows/build.yml/badge.svg)
[![Documentation status](https://github.com/Dzuchun/generic_array_storage/actions/workflows/docs.yml/badge.svg)][docs]
```text
This crate does not require nor std nor alloc. You're welcome ❤️
```

Intertop between [`generic_array`] and [`nalgebra`], for `const usize`-hidden array storages.

This crate will presumably become obsolete, once const generics are introduced, but until then - feel free to unitize it.

**NOTE**: [`nalgebra`]'s storage traits are fundamentally unsafe, so **there is unsafe code inside**, lints mentioned in `Cargo.toml` are there just for extra self-control.

# Adding to a project

```toml
# Cargo.toml
[dependencies]
generic_array_storage = { git = "https://github.com/Dzuchun/generic_array_storage.git", branch = "master" }
```

# Contribution

Just open an issue/PR or something. I'm happy to discuss any additions/fixes!

# Context

## What is `typenum`?

[`typenum`] implements integer operations as types. Basically, it allows for const arithmetic through same sort of trait wizardry, or something 🤷.

The takeaway is:

- if type `A` represents integer `x`
- and type `B` represents integer `y`

then

- `<A typenum::marker_traits::Unsigned>::{U8, U16, .., I8, I16, ..}` are associated constants equal to `x` (if possible)
- (same for `B` and `y`)
- `typenum::operator_aliases::Sum<A, B>` represents integer `x + y`
- `typenum::operator_aliases::Prod<A, B>` represents integer `x * y`

etc

## What is `generic_array`?

[`generic_array`] implements arrays sized via `ArrayLength` trait implementors. Namely, it is implemented for `typenum` types, allowing creation an arrays sized as sum of two other arrays:

```rust
# use generic_array::{sequence::Concat, GenericArray};
// some normal rust arrays
let arr1: [i32; 3] = [1, 2, 3];
let arr2: [i32; 2] = [3, 5];

// some less-normal `generic_array` arrays
// (but having the same size and still stack-allocated)
let garr1 = GenericArray::from_array(arr1);
let garr2 = GenericArray::from_array(arr2);

// array concatenation
let garr_concat = GenericArray::concat(garr1, garr2);

// back to normal rust arrays
let concat: [i32; 5] = garr_concat.into_array();
// let concat: [i32; 6] = garr_concat.into_array(); // <-- does not compile!
```

Coolest thing is - this code is panic-free, fully statically checked, and missized arrays will result in compilation error.

## What is `nalgebra`?

[`nalgebra`] is a matrix manipulation library, abstracted over type actually storing the elements. This allows matrices to be automatically stored on stock, if their dimensions can be inferred at compile-time.

Generally, to store the entire matrix on stack, you'll need for both of it's dimensions to be known, like `nalgrabra::U2` or `nalgebra::U3`. Unfortunately, default storage provided by `nalgebra` has a `const usize` type parameters, so they can't be used in case of sizes provided by associated constants.

## What is `generic_array_storage`?

This crate provides implementation of traits defining `nalgebra` storage backed up by `generic_array` arrays. This allows creation of matrices having dimensions fully expressed as types, completely removing need for `const usize`.

For ease of use, there's a `GenericMatrix` type alias, and `GenericMatrixExt` extension trait, providing convenient type interface and conversion functions respectively. Note that `GenericMatrix` is an alias to `nalgebra::Matrix`, so all of the functions provided by `nalgebra` are expected to be supported.

[`nalgebra`]: https://docs.rs/nalgebra/latest/nalgebra
[`generic_array`]: https://docs.rs/generic-array/latest/generic_array
[`typenum`]: https://docs.rs/typenum/latest/typenum
[docs]: https://dzuchun.github.io/generic_array_storage/generic_array_storage/index.html
