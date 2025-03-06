use generic_array::{ArrayLength, IntoArrayLength};
use nalgebra::{Const, DimName, DimNameAdd, DimNameMul, DimNameProd, DimNameSum, U0, U1, U2};
use typenum::{B0, B1, UInt, UTerm};

/// Convenience trait, used to define type **conv**ersions
///
/// Also, this is the only bound to [`GenericMatrix`](super::GenericMatrix) type alias, meaning that all of the following are valid [`GenericMatrix`](super::GenericMatrix)es:
/// ```rust
/// # use generic_array_storage::GenericMatrix;
/// type NalgebraMatrix = GenericMatrix<i32, nalgebra::U3, nalgebra::U4>;
/// type TypenumMatrix = GenericMatrix<i32, typenum::U3, typenum::U4>;
/// type TypenumConstMatrix = GenericMatrix<i32, typenum::Const<3>, typenum::Const<3>>;
/// // (nalgebra::Const are actually aliased by nalgebra::{U1, U2, ...})
/// ```
pub trait Conv {
    /// [`typenum`]-faced type (unsigned int)
    type TNum: ArrayLength;

    /// [`nalgebra`]-faced type (matrix dimension)
    type Nalg: DimName;

    /// Constructor method used in [`nalgebra`] implementations
    fn new_nalg() -> Self::Nalg {
        Self::Nalg::name()
    }
}

impl Conv for UTerm {
    type Nalg = U0;

    type TNum = Self;
}

impl<U: Conv> Conv for UInt<U, B1>
where
    U: ArrayLength,
    UInt<U, B0>: Conv,
    <UInt<U, B0> as Conv>::Nalg: DimNameAdd<U1>,
{
    type TNum = Self;

    type Nalg = DimNameSum<<UInt<U, B0> as Conv>::Nalg, U1>;
}

impl<U: Conv> Conv for UInt<U, B0>
where
    U: ArrayLength,
    U::Nalg: DimNameMul<U2>,
{
    type TNum = Self;

    type Nalg = DimNameProd<U::Nalg, U2>;
}
type TNum<const N: usize> = typenum::Const<N>;

impl<const N: usize> Conv for TNum<N>
where
    Self: IntoArrayLength,
{
    type TNum = <Self as IntoArrayLength>::ArrayLength;

    type Nalg = Const<N>;
}

impl<const N: usize> Conv for Const<N>
where
    TNum<N>: IntoArrayLength,
{
    type TNum = <TNum<N> as IntoArrayLength>::ArrayLength;

    type Nalg = Self;
}
