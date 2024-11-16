#![doc = include_str!("../README.md")]
#![no_std] // <-- yeah, in case you wondered - there you are, feel free to use it

type TNum<const N: usize> = typenum::Const<N>;
type AlgNum<const N: usize> = nalgebra::Const<N>;
type ArrLen<const N: usize> = <TNum<N> as generic_array::IntoArrayLength>::ArrayLength;
