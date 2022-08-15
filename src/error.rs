macro_rules! create_error_type {
    ($error_name:ident, $format_str:expr, $doc_str:expr) => {
        #[doc=$doc_str]
        #[derive(thiserror::Error, Debug)]
        #[error("{}\n\nsource:\n{}", .msg, .source)]
        pub struct $error_name {
            msg: String,
            #[source]
            source: hdf5::Error
        }


        impl $error_name {
            pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
                let msg = format!($format_str, name);
                Self {msg, source}
            }
        }
    }
}

create_error_type!{
    MissingDataset,
    "failed to read dataset with name `{}`",
    "Error when attempting to read a dataset that does not exist in an hdf5 file"
}

create_error_type!{
    SerializeArray,
    "dataset `{}` exists, but it could not be read to the array type provided",
    r#"Error when attempting to read an array to a given struct type.


This error can occur if the dimension of the h5 array does not match the dimension of the type
provided"#
}

create_error_type!{
    CreateDataset,
    "Failed to create a dataset for `{}`",
    "Could not create a dataset in a hdf5 file when writing"
}

create_error_type!{
    CreateAttribute,
    "Failed to create a attribute for `{}`",
    "Could not create a dataset in a hdf5 file when writing"
}

create_error_type!{
    WriteArray,
    "Failed to write array `{}` to dataset",
    "Failed to serialize an array to a hdf5 dataset"
}

create_error_type!{
    WriteAttribute,
    "Failed to write attribute `{}` to dataset",
    "Failed to serialize an attribute to a hdf5 dataset"
}

create_error_type!{
    FetchDataset,
    "Failed to fetch existing dataset `{}` when writing to file",
    "Could fetch an existing dataset when writing file"
}

create_error_type!{
    FetchAttribute,
    "Failed to fetch existing attribute `{}` when writing to file",
    "Could fetch an existing dataset when writing file"
}

create_error_type!{
    MissingAttribute,
    "Failed to fetch attribute `{}` when reading from file",
    "Could fetch an existing dataset when writing file"
}

create_error_type!{
    SerializeAttribute,
    "Failed to read attribute `{}` to the correct rust type",
    "Failed to serialize the dataset to the correct type after it has been written"
}
