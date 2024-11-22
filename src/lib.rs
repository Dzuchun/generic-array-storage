#![doc = include_str!("../README.md")]
#![no_std] // <-- yeah, in case you wondered - there you are, feel free to use it

use core::fmt::Debug;

use generic_array::{functional::FunctionalSequence, ArrayLength, GenericArray, IntoArrayLength};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, IsContiguous, Matrix, OMatrix, Owned, RawStorage,
    RawStorageMut, Scalar, Storage,
};

mod conv;
pub use conv::Conv;

/// A stack-allocated storage, of [`typenum`]-backed col-major two dimensional array
///
/// This struct is transparent and completely public, since it has nothing to hide! Note that [`GenericArray`] is transparent itself, so this struct effectively has the same layout as a two-dimensional array of the corresponding size.
#[repr(transparent)]
pub struct GenericArrayStorage<T, R: Conv, C: Conv>(
    pub GenericArray<GenericArray<T, R::TNum>, C::TNum>,
);

impl<T: Debug, R: Conv, C: Conv> Debug for GenericArrayStorage<T, R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        <GenericArray<GenericArray<T, R::TNum>, C::TNum> as Debug>::fmt(&self.0, f)
    }
}

impl<T, R: Conv, C: Conv> Clone for GenericArrayStorage<T, R, C>
where
    T: Clone,
    GenericArray<GenericArray<T, R::TNum>, C::TNum>: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T, R: Conv, C: Conv> Copy for GenericArrayStorage<T, R, C>
where
    T: Copy,
    <R::TNum as ArrayLength>::ArrayType<T>: Copy,
    <C::TNum as ArrayLength>::ArrayType<GenericArray<T, R::TNum>>: Copy,
{
}

impl<T, R: Conv, C: Conv> AsRef<[T]> for GenericArrayStorage<T, R, C> {
    fn as_ref(&self) -> &[T] {
        GenericArray::slice_from_chunks(&self.0)
    }
}

impl<T, R: Conv, C: Conv> AsMut<[T]> for GenericArrayStorage<T, R, C> {
    fn as_mut(&mut self) -> &mut [T] {
        GenericArray::slice_from_chunks_mut(&mut self.0)
    }
}

#[allow(unsafe_code, reason = "nalgebra storage traits are unsafe")]
unsafe impl<T, R: Conv, C: Conv> RawStorage<T, R::Nalg, C::Nalg> for GenericArrayStorage<T, R, C> {
    type RStride = nalgebra::U1;

    type CStride = R::Nalg;

    fn ptr(&self) -> *const T {
        if self.0.is_empty() {
            core::ptr::NonNull::<T>::dangling().as_ptr()
        } else {
            self.0.as_ptr().cast()
        }
    }

    fn shape(&self) -> (R::Nalg, C::Nalg) {
        (R::new_nalg(), C::new_nalg())
    }

    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (nalgebra::U1, R::new_nalg())
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    unsafe fn as_slice_unchecked(&self) -> &[T] {
        self.as_ref()
    }
}

#[allow(unsafe_code, reason = "nalgebra storage traits are unsafe")]
unsafe impl<T, R: Conv, C: Conv> RawStorageMut<T, R::Nalg, C::Nalg>
    for GenericArrayStorage<T, R, C>
{
    fn ptr_mut(&mut self) -> *mut T {
        if self.0.is_empty() {
            core::ptr::NonNull::<T>::dangling().as_ptr()
        } else {
            self.0.as_mut_ptr().cast()
        }
    }

    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        // SAFETY: see struct's doc - it's layout is guaranteed to be like that of a two-dimensional array
        self.as_mut()
    }
}

#[allow(unsafe_code, reason = "nalgebra storage traits are unsafe")]
unsafe impl<T: Scalar, R: Conv, C: Conv> Storage<T, R::Nalg, C::Nalg>
    for GenericArrayStorage<T, R, C>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R::Nalg, C::Nalg>,
{
    fn into_owned(self) -> Owned<T, R::Nalg, C::Nalg>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R::Nalg, C::Nalg>,
    {
        nalgebra::DefaultAllocator::allocate_from_iterator(
            R::new_nalg(),
            C::new_nalg(),
            self.0.into_iter().flatten(),
        )
    }

    fn clone_owned(&self) -> Owned<T, R::Nalg, C::Nalg>
    where
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<R::Nalg, C::Nalg>,
    {
        self.clone().into_owned()
    }

    fn forget_elements(self) {
        core::mem::forget(self);
    }
}

#[allow(unsafe_code, reason = "nalgebra storage traits are unsafe")]
unsafe impl<R: Conv, C: Conv, T: nalgebra::Scalar> IsContiguous for GenericArrayStorage<T, R, C> {}

/// Alias to [`nalgebra::Matrix`], completely "hiding" `const usize`s away. See crate's documentation on how this is possible.
pub type GenericMatrix<T, R, C> =
    nalgebra::Matrix<T, <R as Conv>::Nalg, <C as Conv>::Nalg, GenericArrayStorage<T, R, C>>;

type TNum<const N: usize> = typenum::Const<N>;

impl<T, const AR: usize, const AC: usize, R, C> From<[[T; AR]; AC]> for GenericArrayStorage<T, R, C>
where
    TNum<AR>: IntoArrayLength,
    TNum<AC>: IntoArrayLength,
    R: Conv<TNum = <TNum<AR> as IntoArrayLength>::ArrayLength>,
    C: Conv<TNum = <TNum<AC> as IntoArrayLength>::ArrayLength>,
{
    fn from(value: [[T; AR]; AC]) -> Self {
        let tnum_array: GenericArray<
            GenericArray<T, <TNum<AR> as IntoArrayLength>::ArrayLength>,
            <TNum<AC> as IntoArrayLength>::ArrayLength,
        > = GenericArray::from_array(value.map(GenericArray::from_array));
        Self(tnum_array)
    }
}

impl<T, const AR: usize, const AC: usize, R, C> From<GenericArrayStorage<T, R, C>> for [[T; AR]; AC]
where
    TNum<AR>: IntoArrayLength,
    TNum<AC>: IntoArrayLength,
    R: Conv<TNum = <TNum<AR> as IntoArrayLength>::ArrayLength>,
    C: Conv<TNum = <TNum<AC> as IntoArrayLength>::ArrayLength>,
{
    fn from(GenericArrayStorage(data): GenericArrayStorage<T, R, C>) -> Self {
        data.map(GenericArray::into_array).into_array()
    }
}

/// [`GenericMatrix`]-conversion trait intended for core arrays and regular [`nalgebra`] matrices
pub trait GenericMatrixFromExt<R: Conv, C: Conv> {
    /// Type of the elements.
    ///
    /// This an associated type for the simple reason that is can be such.
    type T;

    /// Creates [`GenericMatrix`] from core Rust array.
    fn into_generic_matrix(self) -> GenericMatrix<Self::T, R, C>;
}

impl<T, const AR: usize, const AC: usize, R, C> GenericMatrixFromExt<R, C> for [[T; AR]; AC]
where
    TNum<AR>: IntoArrayLength,
    TNum<AC>: IntoArrayLength,
    R: Conv<TNum = <TNum<AR> as IntoArrayLength>::ArrayLength>,
    C: Conv<TNum = <TNum<AC> as IntoArrayLength>::ArrayLength>,
{
    type T = T;

    fn into_generic_matrix(self) -> GenericMatrix<Self::T, R, C> {
        GenericMatrix::from_data(self.into())
    }
}

impl<T, R, C> GenericMatrixFromExt<R, C> for OMatrix<T, R::Nalg, C::Nalg>
where
    T: Scalar,
    R: Conv,
    C: Conv,
    DefaultAllocator: Allocator<R::Nalg, C::Nalg>,
{
    type T = T;

    fn into_generic_matrix(self) -> GenericMatrix<Self::T, R, C> {
        let (rows, rest) = GenericArray::<_, R::TNum>::chunks_from_slice(self.as_slice());
        debug_assert!(rest.is_empty(), "Should be no leftover");
        let arr = GenericArray::<_, C::TNum>::from_slice(rows);
        let storage = GenericArrayStorage(arr.clone());
        GenericMatrix::from_data(storage)
    }
}

/// Conv trait defining [`GenericMatrix`] conversions.
pub trait GenericMatrixExt {
    /// Type of the elements.
    ///
    /// This an associated type for the simple reason that is can be such.
    type T: Scalar;

    /// Type defining rows count
    type R: Conv;

    /// Type defining column count
    type C: Conv;

    /// Converts [`GenericMatrix`] into regular [`nalgebra`] matrix, backed by core array (it's opaque about that though)
    fn into_regular_matrix(
        self,
    ) -> OMatrix<Self::T, <Self::R as Conv>::Nalg, <Self::C as Conv>::Nalg>
    where
        nalgebra::DefaultAllocator:
            nalgebra::allocator::Allocator<<Self::R as Conv>::Nalg, <Self::C as Conv>::Nalg>;

    /// Changes type of [`GenericMatrix`] to a different row and column count descriptors.
    fn conv<
        NewR: Conv<TNum = <Self::R as Conv>::TNum>,
        NewC: Conv<TNum = <Self::C as Conv>::TNum>,
    >(
        self,
    ) -> GenericMatrix<Self::T, NewR, NewC>;
}

impl<T: Scalar, R: Conv, C: Conv> GenericMatrixExt for GenericMatrix<T, R, C> {
    type T = T;

    type R = R;

    type C = C;

    fn into_regular_matrix(
        self,
    ) -> OMatrix<Self::T, <Self::R as Conv>::Nalg, <Self::C as Conv>::Nalg>
    where
        DefaultAllocator: Allocator<<Self::R as Conv>::Nalg, <Self::C as Conv>::Nalg>,
    {
        Matrix::from_data(DefaultAllocator::allocate_from_iterator(
            <Self::R as Conv>::new_nalg(),
            <Self::C as Conv>::new_nalg(),
            self.data.0.into_iter().flatten(),
        ))
    }

    fn conv<
        NewR: Conv<TNum = <Self::R as Conv>::TNum>,
        NewC: Conv<TNum = <Self::C as Conv>::TNum>,
    >(
        self,
    ) -> GenericMatrix<Self::T, NewR, NewC> {
        GenericMatrix::from_data(GenericArrayStorage(self.data.0))
    }
}

#[cfg(test)]
mod tests;
