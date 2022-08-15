use hdf5_derive::ContainerIo;
use hdf5_derive::HDF5;

#[derive(HDF5)]
struct SimpleAttributeRead {
    #[hdf5(attribute)]
    some_value: u64
}

#[test]
fn simple_attribute_read() {
    let path = "./simple_attribute_read.h5";
    let file = hdf5_derive::File::create(&path).unwrap();

    let value = 999241;

    let attr = file.new_attr::<u64>()
        .create("some_value")
        .unwrap();

    attr.write_scalar(&value).unwrap();

    let x = SimpleAttributeRead::read_hdf5(&file).unwrap();

    assert_eq!(value, x.some_value);

    std::fs::remove_file(&path).unwrap();
}

#[derive(HDF5)]
struct SimpleAttributeWrite {
    #[hdf5(attribute)]
    another_value: u64
}

#[test]
fn simple_attribute_write() {
    let path = "./simple_attribute_write.h5";
    let file = hdf5_derive::File::create(&path).unwrap();

    let value = 21298347;

    let x = SimpleAttributeWrite { another_value: value };
    x.write_hdf5(&file).unwrap();

    let read_value = file.attr("another_value")
        .unwrap()
        .read_scalar()
        .unwrap();

    assert_eq!(value, read_value);

    std::fs::remove_file(&path).unwrap();
}

#[derive(HDF5)]
#[hdf5(mutate_on_write)]
struct MutatedAttributeWrite {
    #[hdf5(attribute)]
    #[hdf5(rename(both="renamed_value"))]
    mutated_value: u64
}

#[test]
fn mutated_attribute_write() {
    let path = "./mutated_attribute_write.h5";
    let file = hdf5_derive::File::create(&path).unwrap();

    let value = 23894;

    let attr = file.new_attr::<u64>()
        .create("renamed_value")
        .unwrap();

    attr.write_scalar(&value).unwrap();

    let new_value = 90842982;
    let mut read_struct = MutatedAttributeWrite::read_hdf5(&file).unwrap();

    assert_eq!(read_struct.mutated_value, value);

    // assign the new value to the struct 
    read_struct.mutated_value = new_value;
    read_struct.write_hdf5(&file).unwrap();

    // then re-read the file and make sure the new value
    let reread_struct = MutatedAttributeWrite::read_hdf5(&file).unwrap();
    assert_eq!(reread_struct.mutated_value, new_value);
    
    std::fs::remove_file(&path).unwrap();
}
