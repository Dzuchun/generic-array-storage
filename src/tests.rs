use nalgebra::{ArrayStorage, Matrix};

use super::{GenericMatrix, GenericMatrixExt};

#[test]
fn from_regular() {
    let regular_matrix = nalgebra::matrix![
        1, 2, 3;
        4, 5, 6;
    ];

    let generic_matrix: GenericMatrix<i32, nalgebra::U2, nalgebra::U3> =
        GenericMatrix::from_regular(regular_matrix);

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
        GenericMatrix::from_array([[1, 2, 5], [3, -4, 0]]);

    let regular_matrix: Matrix<i32, nalgebra::U3, nalgebra::U2, ArrayStorage<i32, 3, 2>> =
        generic_matrix.into_regular();

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
    let a = GenericMatrix::from_regular(nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]);

    let a_tr = a.transpose();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(a[(i, j)], a_tr[(j, i)], "(i, j) = {:?}", (i, j));
        }
    }
}

#[test]
fn to_owned() {
    let a = GenericMatrix::from_regular(nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]);

    assert_eq!(a.clone_owned(), nalgebra::matrix![1, 2, 4; 5, -7, 0]);
}

#[test]
fn addition() {
    let a = GenericMatrix::from_regular(nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]);

    let sum = a + a;

    assert_eq!(sum, nalgebra::matrix![2, 4, 8; 10, -14, 0]);
}

#[test]
fn subtraction() {
    let a = GenericMatrix::from_regular(nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]);

    let sum = a - a;

    assert_eq!(sum, nalgebra::matrix![0, 0, 0; 0, 0, 0]);
}

#[test]
fn multiplication() {
    let a = GenericMatrix::from_regular(nalgebra::matrix![
        1, 2, 4;
        5, -7, 0;
    ]);

    let sum = a * a.transpose();

    assert_eq!(sum, nalgebra::matrix![21, -9; -9, 74]);
}

#[allow(clippy::float_cmp)]
#[test]
fn determinant() {
    let a = GenericMatrix::from_regular(nalgebra::matrix![
        1.0, 3.0;
        -4.0, 4.0;
    ]);

    assert_eq!(a.determinant(), 16.0);
}

#[allow(clippy::float_cmp)]
#[test]
fn inversion() {
    let a = GenericMatrix::from_regular(nalgebra::matrix![
        1.0, 2.0;
        5.0, -7.0;
    ]);

    assert_eq!(
        a.try_inverse().expect("Should be able to inverse"),
        nalgebra::matrix![-7.0, -5.0; -2.0, 1.0].transpose() / -17.0
    );
}
