#![doc = include_str!("../README.md")]
#![no_std] // <-- yeah, in case you wondered - there you are, feel free to use it

type TNum<const N: usize> = typenum::Const<N>;
type AlgNum<const N: usize> = nalgebra::Const<N>;
type ArrLen<const N: usize> = <TNum<N> as generic_array::IntoArrayLength>::ArrayLength;

use generic_array::{ArrayLength, GenericArray, IntoArrayLength};
use nalgebra::{
    allocator::Allocator, ArrayStorage, Dim, IsContiguous, Matrix, Owned, RawStorage, ToTypenum,
};

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

#[allow(unsafe_code, reason = "nalgebra storage traits are unsafe")]
unsafe impl<const R: usize, const C: usize, T> RawStorage<T, AlgNum<R>, AlgNum<C>>
    for GenericArrayStorage<T, ArrLen<R>, ArrLen<C>>
where
    TNum<R>: IntoArrayLength,
    TNum<C>: IntoArrayLength,
    AlgNum<R>: Dim,
    AlgNum<C>: Dim,
{
    type RStride = AlgNum<1>;

    type CStride = AlgNum<R>;

    fn ptr(&self) -> *const T {
        // SAFETY:
        // Trait doc does not state it, but I assume that pointer should be valid and non-null.
        //
        // So this actually returns cast of [`NonNull::dangling`], in case array contains no elements
        if let Some(first) = self.0.first() {
            // there is at least one element - grab it's pointer
            first.as_ptr()
        } else {
            core::ptr::NonNull::<T>::dangling().as_ptr().cast_const()
        }
    }

    fn shape(&self) -> (nalgebra::Const<R>, nalgebra::Const<C>) {
        (nalgebra::Const, nalgebra::Const)
    }

    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (nalgebra::Const, nalgebra::Const)
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    unsafe fn as_slice_unchecked(&self) -> &[T] {
        core::slice::from_raw_parts(
            <Self as nalgebra::RawStorage<T, nalgebra::Const<R>, nalgebra::Const<C>>>::ptr(self),
            R * C,
        )
    }
}

#[allow(unsafe_code, reason = "nalgebra storage traits are unsafe")]
unsafe impl<const R: usize, const C: usize, T>
    nalgebra::RawStorageMut<T, nalgebra::Const<R>, nalgebra::Const<C>>
    for GenericArrayStorage<T, ArrLen<R>, ArrLen<C>>
where
    typenum::Const<R>: IntoArrayLength,
    typenum::Const<C>: IntoArrayLength,
    nalgebra::Const<R>: Dim,
    nalgebra::Const<C>: Dim,
{
    fn ptr_mut(&mut self) -> *mut T {
        self.0
            .first_mut()
            .map_or(core::ptr::NonNull::<T>::dangling().as_ptr(), |first| {
                first.as_mut_ptr()
            })
    }

    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        core::slice::from_raw_parts_mut(
            <Self as nalgebra::RawStorageMut<T, nalgebra::Const<R>, nalgebra::Const<C>>>::ptr_mut(
                self,
            ),
            R * C,
        )
    }
}

#[allow(unsafe_code, reason = "nalgebra storage traits are unsafe")]
unsafe impl<const R: usize, const C: usize, T: nalgebra::Scalar>
    nalgebra::Storage<T, nalgebra::Const<R>, nalgebra::Const<C>>
    for GenericArrayStorage<T, ArrLen<R>, ArrLen<C>>
where
    T: Clone,
    typenum::Const<R>: IntoArrayLength,
    typenum::Const<C>: IntoArrayLength,
    nalgebra::Const<R>: Dim,
    nalgebra::Const<C>: Dim,
{
    fn into_owned(self) -> Owned<T, nalgebra::Const<R>, nalgebra::Const<C>>
    where
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<nalgebra::Const<R>, nalgebra::Const<C>>,
    {
        let init: [[T; R]; C] = self.0.into_array().map(GenericArray::into_array);
        nalgebra::DefaultAllocator::allocate_from_iterator(
            nalgebra::Const::<R>,
            nalgebra::Const::<C>,
            init.into_iter().flatten(),
        )
    }

    fn clone_owned(&self) -> Owned<T, nalgebra::Const<R>, nalgebra::Const<C>>
    where
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<nalgebra::Const<R>, nalgebra::Const<C>>,
    {
        self.clone().into_owned()
    }

    fn forget_elements(self) {
        core::mem::forget(self);
    }
}

#[allow(unsafe_code, reason = "nalgebra storage traits are unsafe")]
unsafe impl<R: ArrayLength, C: ArrayLength, T: nalgebra::Scalar> IsContiguous
    for GenericArrayStorage<T, R, C>
{
}

/// Alias to [`nalgebra::Matrix`], completely "hinding" `const usize`s away. See crate's documentation on how this is possible.
pub type GenericMatrix<T, R, C> = nalgebra::Matrix<
    T,
    R,
    C,
    GenericArrayStorage<
        T,
        <<R as ToTypenum>::Typenum as IntoArrayLength>::ArrayLength,
        <<C as ToTypenum>::Typenum as IntoArrayLength>::ArrayLength,
    >,
>;

impl<T, const R: usize, const C: usize> From<[[T; R]; C]>
    for GenericArrayStorage<
        T,
        <<AlgNum<R> as ToTypenum>::Typenum as IntoArrayLength>::ArrayLength,
        <<AlgNum<C> as ToTypenum>::Typenum as IntoArrayLength>::ArrayLength,
    >
where
    AlgNum<R>: ToTypenum,
    TNum<R>: IntoArrayLength,
    <AlgNum<R> as ToTypenum>::Typenum:
        IntoArrayLength<ArrayLength = <TNum<R> as IntoArrayLength>::ArrayLength>,
    AlgNum<C>: ToTypenum,
    TNum<C>: IntoArrayLength,
    <AlgNum<C> as ToTypenum>::Typenum:
        IntoArrayLength<ArrayLength = <TNum<C> as IntoArrayLength>::ArrayLength>,
{
    fn from(value: [[T; R]; C]) -> Self {
        GenericArrayStorage(GenericArray::from_array(
            value.map(GenericArray::from_array),
        ))
    }
}

impl<T, const R: usize, const C: usize>
    From<
        GenericArrayStorage<
            T,
            <<AlgNum<R> as ToTypenum>::Typenum as IntoArrayLength>::ArrayLength,
            <<AlgNum<C> as ToTypenum>::Typenum as IntoArrayLength>::ArrayLength,
        >,
    > for [[T; R]; C]
where
    AlgNum<R>: ToTypenum,
    TNum<R>: IntoArrayLength,
    <AlgNum<R> as ToTypenum>::Typenum:
        IntoArrayLength<ArrayLength = <TNum<R> as IntoArrayLength>::ArrayLength>,
    AlgNum<C>: ToTypenum,
    TNum<C>: IntoArrayLength,
    <AlgNum<C> as ToTypenum>::Typenum:
        IntoArrayLength<ArrayLength = <TNum<C> as IntoArrayLength>::ArrayLength>,
{
    fn from(
        value: GenericArrayStorage<
            T,
            <<AlgNum<R> as ToTypenum>::Typenum as IntoArrayLength>::ArrayLength,
            <<AlgNum<C> as ToTypenum>::Typenum as IntoArrayLength>::ArrayLength,
        >,
    ) -> Self {
        value.0.into_array().map(GenericArray::into_array)
    }
}
