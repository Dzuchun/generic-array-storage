use crate::{GenericMatrix, GenericMatrixExt, GenericMatrixFromExt};

#[test]
fn from_regular() {
    let regular_matrix = nalgebra::matrix![
        1, 2, 3;
        4, 5, 6;
    ];

    let generic_matrix: GenericMatrix<i32, nalgebra::U2, nalgebra::U3> =
        regular_matrix.into_generic_matrix();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(
                generic_matrix[(i, j)],
                regular_matrix[(i, j)],
                "(i, j) = {:?}",
                (i, j)
            );
        }
    }
}

#[test]
fn into_regular() {
    let generic_matrix: GenericMatrix<i32, nalgebra::U3, nalgebra::U2> =
        [[1, 2, 5], [3, -4, 0]].into_generic_matrix();

    let regular_matrix = generic_matrix.into_regular_matrix();

    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(
                generic_matrix[(i, j)],
                regular_matrix[(i, j)],
                "(i, j) = {:?}",
                (i, j)
            );
        }
    }
}

#[test]
fn transposition() {
    let a: GenericMatrix<_, nalgebra::U2, nalgebra::U3> = nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]
    .into_generic_matrix();

    let a_tr = a.transpose();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(a[(i, j)], a_tr[(j, i)], "(i, j) = {:?}", (i, j));
        }
    }
}

#[test]
fn to_owned() {
    let a: GenericMatrix<_, typenum::Const<2>, typenum::Const<3>> = nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]
    .into_generic_matrix();

    assert_eq!(a.clone_owned(), nalgebra::matrix![1, 2, 4; 5, -7, 0]);
}

#[test]
fn addition() {
    let a: GenericMatrix<_, typenum::U2, typenum::U3> = nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]
    .into_generic_matrix();

    let sum = a + a;

    assert_eq!(sum, nalgebra::matrix![2, 4, 8; 10, -14, 0]);
}

#[test]
fn subtraction() {
    let a: GenericMatrix<_, nalgebra::Const<2>, nalgebra::Const<3>> = nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]
    .into_generic_matrix();

    let sum = a - a;

    assert_eq!(sum, nalgebra::matrix![0, 0, 0; 0, 0, 0]);
}

#[test]
fn multiplication() {
    let a: GenericMatrix<_, nalgebra::Const<2>, nalgebra::Const<3>> = nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]
    .into_generic_matrix();

    let sum = a * a.transpose();

    assert_eq!(sum, nalgebra::matrix![21, -9; -9, 74]);
}

#[allow(clippy::float_cmp)]
#[test]
fn determinant() {
    let a: GenericMatrix<_, typenum::U2, typenum::U2> = nalgebra::matrix![
        1.0, 3.0;
        -4.0, 4.0;
    ]
    .into_generic_matrix();

    assert_eq!(a.determinant(), 16.0);
}

#[allow(clippy::float_cmp)]
#[test]
fn inversion() {
    let a: GenericMatrix<_, typenum::U2, typenum::U2> = nalgebra::matrix![
        1.0, 3.0;
        -4.0, 4.0;
    ]
    .into_generic_matrix();

    assert_eq!(
        a.try_inverse().expect("Should be able to inverse"),
        nalgebra::matrix![4.0, 4.0; -3.0, 1.0].transpose() / 16.0
    );
}

#[allow(
    unused_variables,
    unused_assignments,
    reason = "Test is the compilation"
)]
#[test]
fn comp() {
    let data = [[1, 2, 3], [-4, 5, -0], [-232, 343, 232_111]];

    let mut typenum_matrix: GenericMatrix<_, typenum::U3, typenum::U3> = data.into_generic_matrix();
    let mut nalgebra_matrix: GenericMatrix<_, nalgebra::U3, nalgebra::U3> =
        data.into_generic_matrix();
    let mut typenum_const_matrix: GenericMatrix<_, typenum::Const<3>, typenum::Const<3>> =
        data.into_generic_matrix();
    // nalgebra's const and alias types are identical

    // note, that none of them can be directly assigned to each other:
    // typenum_matrix = nalgebra_matrix;
    // typenum_const_matrix = typenum_matrix;
    // nalgebra_matrix = typenum_const_matrix;
    // (all of the above lines do fail)

    // but assignments will succeed, once conversion is used:
    typenum_matrix = nalgebra_matrix.conv();
    typenum_const_matrix = typenum_matrix.conv();
    nalgebra_matrix = typenum_const_matrix.conv();

    // of there's matrix of different dimensions,
    let some_different_matrix: GenericMatrix<_, typenum::U1, typenum::U3> =
        nalgebra::matrix![1, 2, 3].into_generic_matrix();

    // all of the assignments fail:
    // typenum_matrix = some_different_matrix;
    // nalgebra_matrix = some_different_matrix;
    // typenum_const_matrix = some_different_matrix;
    // (all of the above lines do fail)
}
