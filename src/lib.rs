#![doc = include_str!("../README.md")]
#![no_std] // <-- yeah, in case you wondered - there you are, feel free to use it

/// Nothing to see here
///
/// Congratulations, you've found a cute seal pup!
///
/// <img src="https://www.startpage.com/av/proxy-image?piurl=https%3A%2F%2Fi.pinimg.com%2Foriginals%2Fb8%2F2f%2F61%2Fb82f6130b6a8d6afbefc07b08e1f8750.jpg&sp=1731807377Td567a79c382f033584e92b2a748b4f2c5e21b672386c3919dd66700457951351" alt="cute baby seal!" width="10%"/>
///
/// This tiny thing ensures that [`Conv`], [`Corr`] and [`Comp`] traits are not implemented outside of this crate. There's no strict reasoning for this now, but I might want to use some **unsafe** operations in these trait implementations in the future.
#[derive(Debug)]
pub struct Seal(());

/// Convenience trait, used to defining type **conv**ersions
///
/// Also, this is the only bound to [`GenericMatrix`] type alias, meaning that all of the following are valid [`GenericMatrix`]es:
/// ```rust
/// # use generic_array_storage::GenericMatrix;
/// type NalgebraMatrix = GenericMatrix<i32, nalgebra::U3, nalgebra::U4>;
/// type TypenumMatrix = GenericMatrix<i32, typenum::U3, typenum::U4>;
/// type TypenumConstMatrix = GenericMatrix<i32, typenum::Const<3>, typenum::Const<3>>;
/// // (nalgebra::Const are actually aliased by nalgebra::{U1, U2, ...})
/// ```
///
/// If you need to convert between them, see [`Comp`].
///
/// This trait is sealed.
pub trait Conv {
    /// See [`Seal`]
    const SEAL: Seal;

    /// A core `usize` corresponding to length/size
    const NUM: usize;

    /// [`nalgebra`]-faced type (matrix dimension)
    type Nalg: Dim;

    /// [`typenum`]-faced type (unsigned int)
    type TNum: Unsigned;

    /// [`generic_array`]-faced type (generic array length)
    type ArrLen: ArrayLength;

    /// Constructor method used in [`nalgebra`] implementations
    fn new_nalg() -> Self::Nalg;
}

impl<const N: usize> Conv for nalgebra::Const<N>
where
    typenum::Const<N>: generic_array::IntoArrayLength + typenum::ToUInt,
    <typenum::Const<N> as typenum::ToUInt>::Output: Unsigned,
{
    const SEAL: Seal = Seal(());

    const NUM: usize = N;

    type Nalg = nalgebra::Const<N>;

    type TNum = typenum::U<N>;

    type ArrLen = <typenum::Const<N> as generic_array::IntoArrayLength>::ArrayLength;

    fn new_nalg() -> Self::Nalg {
        nalgebra::Const::<N>
    }
}

impl<const N: usize> Conv for typenum::Const<N>
where
    typenum::Const<N>: generic_array::IntoArrayLength + typenum::ToUInt,
    <typenum::Const<N> as typenum::ToUInt>::Output: Unsigned,
{
    const SEAL: Seal = Seal(());

    const NUM: usize = N;

    type Nalg = nalgebra::Const<N>;

    type TNum = typenum::U<N>;

    type ArrLen = <typenum::Const<N> as generic_array::IntoArrayLength>::ArrayLength;

    fn new_nalg() -> Self::Nalg {
        nalgebra::Const::<N>
    }
}

impl Conv for typenum::UTerm {
    const SEAL: Seal = Seal(());

    const NUM: usize = 0;

    type Nalg = nalgebra::Const<0>;

    type TNum = typenum::consts::U0;

    type ArrLen = Self;

    fn new_nalg() -> Self::Nalg {
        nalgebra::Const::<0>
    }
}

impl<U> Conv for typenum::UInt<U, B0>
where
    U: Conv,
    U::Nalg: nalgebra::dimension::DimMul<nalgebra::U2>,
    U::ArrLen: core::ops::Mul<typenum::U2>,
    typenum::Prod<U::ArrLen, typenum::U2>: ArrayLength,
{
    const SEAL: Seal = Seal(());

    const NUM: usize = 2 * U::NUM;

    type Nalg = <U::Nalg as nalgebra::dimension::DimMul<nalgebra::U2>>::Output;

    type TNum = typenum::operator_aliases::Prod<U::ArrLen, typenum::U2>;

    type ArrLen = typenum::operator_aliases::Prod<U::ArrLen, typenum::U2>;

    fn new_nalg() -> Self::Nalg {
        Self::Nalg::from_usize(Self::NUM)
    }
}

impl<U> Conv for typenum::UInt<U, B1>
where
    typenum::UInt<U, B0>: Conv,
    <typenum::UInt<U, B0> as Conv>::Nalg: nalgebra::dimension::DimAdd<nalgebra::U1>,
    <typenum::UInt<U, B0> as Conv>::ArrLen: core::ops::Add<typenum::U1>,
    typenum::Sum<<typenum::UInt<U, B0> as Conv>::ArrLen, typenum::U1>: ArrayLength,
{
    const SEAL: Seal = Seal(());

    const NUM: usize = 1 + <typenum::UInt<U, B0> as Conv>::NUM;

    type Nalg =
        <<typenum::UInt<U, B0> as Conv>::Nalg as nalgebra::dimension::DimAdd<nalgebra::U1>>::Output;

    type TNum = typenum::operator_aliases::Sum<<typenum::UInt<U, B0> as Conv>::ArrLen, typenum::U1>;

    type ArrLen =
        typenum::operator_aliases::Sum<<typenum::UInt<U, B0> as Conv>::ArrLen, typenum::U1>;

    fn new_nalg() -> Self::Nalg {
        Self::Nalg::from_usize(Self::NUM)
    }
}

/// Convenience trait, used to signify that particular [`Conv`] implementor **corr**esponds to some integer. This has to be a separate trait, since integer correspondence is not something we always need, and the actual point of this crate is to get rid of such bounds. This trait is a way to reintroduce it, if you happen to need that.
///
/// Trait defines methods for core array to/from [`GenericArray`] conversion. Implementations are expected to be trivial, i.e. [`GenericArray`] method call or unsafe transmute backed by some explanation.
///
/// This trait is sealed.
pub trait Corr<const N: usize>: Conv {
    /// See [`Seal`]
    const SEAL: Seal;

    /// Converts core array to [`GenericArray`]
    fn array_to_generic<E>(array: [E; N]) -> GenericArray<E, Self::ArrLen>;

    /// Converts [`GenericArray`] to core array
    fn generic_to_array<E>(generic: GenericArray<E, Self::ArrLen>) -> [E; N];
}

impl<const N: usize, T> Corr<N> for T
where
    T: Conv,
    typenum::Const<N>: IntoArrayLength<ArrayLength = T::ArrLen>,
{
    const SEAL: Seal = Seal(());

    #[inline]
    fn array_to_generic<E>(array: [E; N]) -> GenericArray<E, Self::ArrLen> {
        GenericArray::from_array(array)
    }

    #[inline]
    fn generic_to_array<E>(generic: GenericArray<E, Self::ArrLen>) -> [E; N] {
        generic.into_array()
    }
}

/// Convenience trait, used to signify that a pair of [`Conv`] implementors are **comp**atible with each other, meaning that [`GenericArray`]s sized with them can be converted [`Comp::fwd`] and [`Comp::bwd`].
///
/// This trait is sealed.
pub trait Comp<Other: Conv>: Conv {
    /// See [`Seal`]
    const SEAL: Seal;

    /// "forward" conversion
    fn fwd<E>(value: GenericArray<E, Self::ArrLen>) -> GenericArray<E, Other::ArrLen>;

    /// "backward" conversion
    fn bwd<E>(value: GenericArray<E, Other::ArrLen>) -> GenericArray<E, Self::ArrLen>;
}

impl<T: Conv, Other: Conv> Comp<Other> for T
where
    T: Conv<ArrLen = Other::ArrLen>,
{
    const SEAL: Seal = Seal(());

    #[inline]
    fn fwd<E>(value: GenericArray<E, Self::ArrLen>) -> GenericArray<E, <Other as Conv>::ArrLen> {
        value
    }

    #[inline]
    fn bwd<E>(value: GenericArray<E, <Other as Conv>::ArrLen>) -> GenericArray<E, Self::ArrLen> {
        value
    }
}

use generic_array::{functional::FunctionalSequence, ArrayLength, GenericArray, IntoArrayLength};
use nalgebra::{
    allocator::Allocator, ArrayStorage, Dim, IsContiguous, Matrix, Owned, RawStorage,
    RawStorageMut, Scalar, Storage,
};
use typenum::{Unsigned, B0, B1};

/// A stack-allocated storage, of [`typenum`]-backed col-major two dimensional array
///
/// This struct is transparent and completely public, since it has nothing to hide! Note that [`GenericArray`] is transparent itself, so this struct effectively has the same layout as a two-dimensional array of the corresponding size.
#[derive(Debug)]
#[repr(transparent)]
pub struct GenericArrayStorage<T, R: Conv, C: Conv>(
    pub GenericArray<GenericArray<T, R::ArrLen>, C::ArrLen>,
);

impl<T, R: Conv, C: Conv> Clone for GenericArrayStorage<T, R, C>
where
    T: Clone,
    GenericArray<GenericArray<T, R::ArrLen>, C::ArrLen>: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T, R: Conv, C: Conv> Copy for GenericArrayStorage<T, R, C>
where
    T: Copy,
    <R::ArrLen as ArrayLength>::ArrayType<T>: Copy,
    <C::ArrLen as ArrayLength>::ArrayType<GenericArray<T, R::ArrLen>>: Copy,
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

impl<T, const AR: usize, const AC: usize, R: Conv + Corr<AR>, C: Conv + Corr<AC>>
    From<[[T; AR]; AC]> for GenericArrayStorage<T, R, C>
{
    fn from(value: [[T; AR]; AC]) -> Self {
        GenericArrayStorage(C::array_to_generic(value.map(R::array_to_generic)))
    }
}

impl<T, const AR: usize, const AC: usize, R: Conv + Corr<AR>, C: Conv + Corr<AC>>
    From<GenericArrayStorage<T, R, C>> for [[T; AR]; AC]
{
    fn from(value: GenericArrayStorage<T, R, C>) -> Self {
        C::generic_to_array(value.0).map(R::generic_to_array)
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

impl<T, const AR: usize, const AC: usize, R: Conv + Corr<AR>, C: Conv + Corr<AC>>
    GenericMatrixFromExt<R, C> for [[T; AR]; AC]
{
    type T = T;

    fn into_generic_matrix(self) -> GenericMatrix<Self::T, R, C> {
        GenericMatrix::from_data(self.into())
    }
}

// Yes, I AM sorry for the `T: Clone` here.
impl<T: Clone, R: Conv, C: Conv, S: RawStorage<T, R::Nalg, C::Nalg> + IsContiguous>
    GenericMatrixFromExt<R, C> for Matrix<T, R::Nalg, C::Nalg, S>
{
    type T = T;

    fn into_generic_matrix(self) -> GenericMatrix<Self::T, R, C> {
        let (rows, rest) = GenericArray::<_, R::ArrLen>::chunks_from_slice(self.as_slice());
        debug_assert!(rest.is_empty(), "Should be no leftover");
        let arr = GenericArray::<_, C::ArrLen>::from_slice(rows);
        let storage = GenericArrayStorage(arr.clone());
        GenericMatrix::from_data(storage)
    }
}

/// Convenience trait defining [`GenericMatrix`] conversions.
pub trait GenericMatrixExt {
    /// Type of the elements.
    ///
    /// This an associated type for the simple reason that is can be such.
    type T;

    /// Type defining rows count
    type R: Conv;

    /// Type defining column count
    type C: Conv;

    /// Changes type of [`GenericMatrix`] to a different row and column count descriptors.
    fn conv<NewR: Conv + Comp<Self::R>, NewC: Conv + Comp<Self::C>>(
        self,
    ) -> GenericMatrix<Self::T, NewR, NewC>;

    /// Converts [`GenericMatrix`] into core array-backed regular [`nalgebra`] matrix
    fn into_array_matrix<const AR: usize, const AC: usize>(
        self,
    ) -> Matrix<
        Self::T,
        nalgebra::Const<AR>,
        nalgebra::Const<AC>,
        nalgebra::ArrayStorage<Self::T, AR, AC>,
    >
    where
        Self::R: Corr<AR>,
        Self::C: Corr<AC>;
}

impl<T, R: Conv, C: Conv> GenericMatrixExt for GenericMatrix<T, R, C> {
    type T = T;

    type R = R;

    type C = C;

    fn conv<NewR: Conv + Comp<Self::R>, NewC: Conv + Comp<Self::C>>(
        self,
    ) -> GenericMatrix<Self::T, NewR, NewC> {
        let data = self.data.0;
        let mapped_cols = data.map(NewR::bwd);
        let full_mapped = NewC::bwd(mapped_cols);
        GenericMatrix::from_data(GenericArrayStorage(full_mapped))
    }

    fn into_array_matrix<const AR: usize, const AC: usize>(
        self,
    ) -> Matrix<
        Self::T,
        nalgebra::Const<AR>,
        nalgebra::Const<AC>,
        nalgebra::ArrayStorage<Self::T, AR, AC>,
    >
    where
        Self::R: Corr<AR>,
        Self::C: Corr<AC>,
    {
        let data = self.data;
        let array: [[Self::T; AR]; AC] = data.into();
        Matrix::from_data(ArrayStorage(array))
    }
}

#[cfg(test)]
mod tests;
