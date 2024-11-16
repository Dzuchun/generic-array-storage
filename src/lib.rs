#![doc = include_str!("../README.md")]
#![no_std] // <-- yeah, in case you wondered - there you are, feel free to use it

type TNum<const N: usize> = typenum::Const<N>;
type AlgNum<const N: usize> = nalgebra::Const<N>;
type ArrLen<const N: usize> = <TNum<N> as generic_array::IntoArrayLength>::ArrayLength;

use generic_array::{ArrayLength, GenericArray};
/// A stack-allocated storage, of [`typenum`]-backed row-major two dimensional array
#[derive(Debug, Clone)]
pub struct GenericArrayStorage<T, R, C>(GenericArray<GenericArray<T, R>, C>)
where
    R: ArrayLength,
    C: ArrayLength;

impl<T, R, C> Copy for GenericArrayStorage<T, R, C>
where
    T: Copy,
    R: ArrayLength,
    R::ArrayType<T>: Copy,
    C: ArrayLength,
    C::ArrayType<GenericArray<T, R>>: Copy,
{
}
