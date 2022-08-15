use hdf5_derive::ContainerIo;
use hdf5_derive::File;
use macros::HDF5;
use std::fs;

type Arr3 = ndarray::Array3<f64>;
use ndarray::Array2;

#[derive(HDF5)]
struct TestWrite {
    one: Arr3,
}

#[test]
fn simple_write_array() {
    let path = "simple_write_array.h5";
    fs::remove_file(path).ok();
    let file = File::create(path).unwrap();

    let shape = (5, 20, 20);
    let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
        .into_shape(shape)
        .unwrap();

    let x = TestWrite { one: arr.clone() };

    x.write_hdf5(&file).unwrap();

    let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

    // check the arrays are the same
    assert_eq!(x.one, new_arr);

    fs::remove_file(path).ok();
}

#[derive(HDF5)]
struct TransposeWrite {
    #[hdf5(transpose = "write")]
    one: Arr3,
}

#[test]
fn write_transposed() {
    let path = "write_transposed_1.h5";
    fs::remove_file(path).ok();
    let file = File::create(path).unwrap();

    let shape = (5, 20, 20);
    let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
        .into_shape(shape)
        .unwrap();

    let x = TransposeWrite { one: arr.clone() };

    x.write_hdf5(&file).unwrap();

    let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

    // check the arrays were transposed
    assert_eq!(x.one.t(), new_arr);

    fs::remove_file(path).ok();
}

#[derive(HDF5)]
#[hdf5(transpose = "write")]
struct TransposeWriteInherit {
    one: Arr3,
}

#[test]
fn write_transposed_inherit() {
    let path = "write_transposed_inherit.h5";
    fs::remove_file(path).ok();
    let file = File::create(path).unwrap();

    let shape = (5, 20, 20);
    let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
        .into_shape(shape)
        .unwrap();

    let x = TransposeWriteInherit { one: arr.clone() };

    x.write_hdf5(&file).unwrap();

    let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

    // check the arrays were transposed
    assert_eq!(x.one.t(), new_arr);

    fs::remove_file(path).ok();
}

#[derive(HDF5)]
#[hdf5(transpose = "write")]
struct TransposeWriteOverride {
    #[hdf5(transpose = "none")]
    one: Arr3,
}

#[test]
fn write_transposed_override() {
    let path = "write_transposed_override.h5";
    fs::remove_file(path).ok();
    let file = File::create(path).unwrap();

    let shape = (5, 20, 20);
    let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
        .into_shape(shape)
        .unwrap();

    let x = TransposeWriteOverride { one: arr.clone() };

    x.write_hdf5(&file).unwrap();

    let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

    // check the arrays are the same
    assert_eq!(x.one, new_arr);

    fs::remove_file(path).ok();
}

#[derive(HDF5)]
struct ManuallyTransposedArray {
    #[hdf5(transpose = "write")]
    one: Arr3,
}

#[test]
/// ensure non-standard layout arrays (non-contiguous) can be written without errors
fn manually_transposed_array() {
    let path = "manually_transposed_array.h5";
    fs::remove_file(path).ok();
    let file = File::create(path).unwrap();

    let shape = (5, 4, 4);

    // write data out
    let arr = ndarray::Array::linspace(
        0.,
        (shape.0 * shape.1 * shape.2) as f64 - 1.,
        shape.0 * shape.1 * shape.2,
    )
    .into_shape(shape)
    .unwrap();

    // transpose the data outside of the library
    let arr = arr.t().to_owned();

    // the array is not standard layout
    assert!(arr.is_standard_layout() == false);

    let x = ManuallyTransposedArray { one: arr.clone() };

    x.write_hdf5(&file).unwrap();

    let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

    // check the arrays are the same
    assert_eq!(arr.t(), new_arr);

    fs::remove_file(path).ok();
}

#[derive(HDF5)]
#[hdf5(mutate_on_write=true)]
struct MutateOnWrite {
    one: Arr3,
}

#[test]
fn mutate_on_write() {
    let path = "mutate_on_write.h5";
    fs::remove_file(path).ok();
    let file = File::create(path).unwrap();

    let shape = (5, 4, 4);

    // write data out
    let arr = ndarray::Array::linspace(
        0.,
        (shape.0 * shape.1 * shape.2) as f64 - 1.,
        shape.0 * shape.1 * shape.2,
    )
    .into_shape(shape)
    .unwrap();

     // create an existing dataset so we will error in .write_hdf5()
     // if we try to create it without mutating
     file
        .new_dataset::<f64>()
        .shape(shape)
        .create("one")
        .unwrap();

    let x = MutateOnWrite { one: arr.clone() };

    x.write_hdf5(&file).unwrap();

    let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

    // check the arrays are the same
    assert_eq!(arr, new_arr);

    fs::remove_file(path).ok();
}

#[derive(HDF5)]
#[hdf5(mutate_on_write=true)]
struct MutateOnWriteDifferentShapes {
    one: Array2<usize>,
}

#[derive(HDF5)]
struct DifferentShapes {
    one: Array2<usize>,
}

#[test]
fn mutate_on_write_different_shapes() {
    let path = "mutate_on_write_different_shapes.h5";
    let file = hdf5::File::create(&path).unwrap();

    let arr = Array2::zeros((5,4));
    let arr_another = Array2::zeros((3,2));
    
    let h5_writer = DifferentShapes {one : arr.clone() };
    h5_writer.write_hdf5(&file).unwrap();

    // then read the data into a struct that mutates it when it writes
    let mut mutate_writer = MutateOnWriteDifferentShapes::read_hdf5(&file).unwrap();
    mutate_writer.one = arr_another.clone();
    let out = mutate_writer.write_hdf5(&file);

    assert!(out.is_err());

    fs::remove_file(path).ok();
}
