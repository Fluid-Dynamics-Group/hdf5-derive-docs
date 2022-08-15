#![doc = include_str!("../README.md")]

/// Derive read and write capabilities for a struct of arrays
///
/// refer to the [crate-level](index.html) documentation
pub use macros::HDF5;

pub use hdf5::File;
pub use hdf5::Group;

pub mod error;

#[doc(hidden)]
pub use error::*;

/// Provides methods for reading and writing to an [`hdf5`] file. Derived with [`HDF5`] macro.
pub trait ContainerIo {
    /// write the contents of a struct to an HDF5 file
    ///
    ///
    /// ```
    /// use hdf5_derive::ContainerIo;
    /// use hdf5_derive::HDF5;
    /// use ndarray::Array2;
    ///
    /// #[derive(HDF5)]
    /// struct Data {
    ///     some_field: Array2<u32>
    /// }
    ///
    /// let path = "./test_file_write.h5";
    /// let file = hdf5_derive::File::create(path).unwrap();
    ///
    /// // write some data to the file
    /// let arr = Array2::zeros((5,5));
    /// let data = Data { some_field: arr.clone() };
    /// data.write_hdf5(&file).unwrap();
    ///
    /// // manually read the data using `hdf5` primitives
    /// let dset = file.dataset("some_field").unwrap();
    /// let read_array : Array2<u32> = dset.read().unwrap();
    ///
    /// // check that they are the same
    /// assert_eq!(read_array, arr);
    ///
    /// // remove this file for practical purposes
    /// std::fs::remove_file(path).unwrap();
    /// ```
    fn write_hdf5(&self, container: &File) -> Result<(), Error>;
    /// read the contents of an HDF5 file to `Self`
    ///
    /// ```
    /// use hdf5_derive::ContainerIo;
    /// use hdf5_derive::HDF5;
    /// use ndarray::Array2;
    ///
    /// #[derive(HDF5)]
    /// struct Data {
    ///     #[hdf5(rename(read = "some_field_renamed"))]
    ///     some_field: Array2<u32>
    /// }
    ///
    /// let path = "./test_file_read.h5";
    /// let file = hdf5_derive::File::create(path).unwrap();
    ///
    /// // write some data to the file
    /// let arr = Array2::zeros((5,5));
    /// let dset = file.new_dataset::<u32>()
    ///     .shape((5,5))
    ///     .create("some_field_renamed")
    ///     .unwrap();
    /// dset.write(&arr).unwrap();
    ///
    /// // now, read the data from the written dataset
    /// let read_data = Data::read_hdf5(&file).unwrap();
    ///
    /// // check that they are the same
    /// assert_eq!(read_data.some_field, arr);
    ///
    /// // remove this file for practical purposes
    /// std::fs::remove_file(path).unwrap();
    /// ```
    fn read_hdf5(container: &Group) -> Result<Self, Error>
    where
        Self: Sized;
}

#[derive(thiserror::Error, Debug)]
/// General error type that provides helpful information on what went wrong
pub enum Error {
    #[error(transparent)]
    /// Error when attempting to read a dataset that does not exist in an hdf5 file
    MissingDataset(#[from] error::MissingDataset),
    #[error(transparent)]
    /// Error when attempting to read an array to a given struct type
    ///
    /// This error can occur if the dimension of the h5 array does not match the dimension of the type
    /// provided
    SerializeArray(#[from] error::SerializeArray),
    #[error(transparent)]
    /// Failed to serialize an array to a hdf5 dataset
    WriteArray(#[from] error::WriteArray),
    #[error(transparent)]
    /// Could not create a dataset in a hdf5 file when writing
    CreateDataset(#[from] error::CreateDataset),
    #[error(transparent)]
    /// Could fetch an existing dataset when writing file
    FetchDataset(#[from] error::FetchDataset),
    #[error(transparent)]
    /// Attribute was missing from hdf5 file
    MissingAttribute(#[from] error::MissingAttribute),
    #[error(transparent)]
    /// Attribute was missing from hdf5 file
    SerializeAttribute(#[from] error::SerializeAttribute),
    #[error(transparent)]
    /// Attribute was missing from hdf5 file
    FetchAttribute(#[from] error::FetchAttribute),
    #[error(transparent)]
    /// Could not create a attribute in a hdf5 file when writing
    CreateAttribute(#[from] error::CreateAttribute),
    #[error(transparent)]
    /// Could not create a attribute in a hdf5 file when writing
    WriteAttribute(#[from] error::WriteAttribute),
}

/// Helper trait to determine the type of element that a given [`ArrayBase`](ndarray::ArrayBase)
/// implements
///
/// This trait is required because `hdf5` datasets require type information on the data they store
/// before creation, and the type system is not smart enough to determine this type based on the
/// subsequent writes.
///
/// Since the `ArrayBase` does not directly implement [`RawData`](ndarray::RawData), it is not
/// possible to determine the type of the array elements without a helper trait.
#[doc(hidden)]
pub trait ArrayType {
    type Ty;
}

impl<S, D> ArrayType for ndarray::ArrayBase<S, D>
where
    S: ndarray::RawData,
{
    type Ty = <S as ndarray::RawData>::Elem;
}
