mod read;
mod write;

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput, Result};
use syn::spanned::Spanned;
use darling::{ast, FromDeriveInput, FromField};

#[proc_macro_derive(HDF5, attributes(hdf5))]
pub fn hdf5(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    derive(input)
        .map(Into::into)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}


#[derive(Debug, Clone, Copy, darling::FromMeta)]
#[darling(default)]
enum TransposeOpts {
    Read,
    Write,
    Both,
    None
}

impl TransposeOpts {
    fn transpose_read(&self) -> bool {
        matches!(self, Self::Read | Self::Both)
    }

    fn transpose_write(&self) -> bool {
        matches!(self, Self::Write | Self::Both)
    }
}

impl Default for TransposeOpts{
    fn default() -> Self {
        TransposeOpts::None
    }
}

#[derive(Debug, Clone, darling::FromMeta, Default)]
#[darling(default)]
struct Rename {
    read: Option<String>,
    write: Option<String>,
    both: Option<String>,
}

impl Rename {
    /// name of the array if we are reading
    fn read_name_or_ident(&self, ident: &syn::Ident) -> String {
        self.both.as_ref().or(self.read.as_ref()).map(Into::into).unwrap_or_else(|| ident.to_string())
    }

    /// name of the array if we are writing
    fn write_name_or_ident(&self, ident: &syn::Ident) -> String {
        self.both.as_ref().or(self.write.as_ref()).map(Into::into).unwrap_or_else( || ident.to_string())
    }
}

#[derive(Debug, FromDeriveInput)]
#[darling(supports(struct_any), attributes(hdf5))]
struct InputReceiver {
    /// The struct ident.
    #[allow(dead_code)]
    ident: syn::Ident,

    #[allow(dead_code)]
    generics: syn::Generics,

    /// Receives the body of the struct or enum. We don't care about
    /// struct fields because we previously told darling we only accept structs.
    data: ast::Data<(), FieldReceiver>,

    #[darling(default)]
    transpose: TransposeOpts,

    #[darling(default)]
    mutate_on_write: bool,
}

#[derive(Debug, FromField)]
#[darling(attributes(hdf5))]
struct FieldReceiver {
    /// Get the ident of the field. For fields in tuple or newtype structs or
    /// enum bodies, this can be `None`.
    ident: Option<syn::Ident>,

    /// This magic field name pulls the type from the input.
    ty: syn::Type,

    #[darling(default)]
    /// whether or not to use `std::ops::Deref` on the field before 
    /// serializing the container
    transpose: Option<TransposeOpts>,

    #[darling(default)]
    /// whether or not to use `std::ops::Deref` on the field before 
    /// serializing the container
    rename: Rename,

    #[darling(default)]
    mutate_on_write: Option<bool>,


    #[darling(default)]
    #[darling(rename = "attribute")]
    is_attribute: bool
}

fn derive(input: DeriveInput) -> Result<TokenStream> {
    let receiver = InputReceiver::from_derive_input(&input).unwrap();

    // make sure we are dealing with a non-tuple / unit struct  struct
    let fields_information = match receiver.data {
        ast::Data::Enum(_) => unreachable!(),
        ast::Data::Struct(fields_with_style) => {
            match fields_with_style.style {
                ast::Style::Tuple | ast::Style::Unit => {
                    Err(
                        syn::parse::Error::new(input.span(), "Tuple / Unit structs are not accepted. Each field must be named")
                    )
                }
                ast::Style::Struct => {
                    Ok(
                        fields_with_style.fields
                    )
                }
            }
        }
    }?;

    // build the reading body:
    let read_data: Vec<read::ReadInfo> = fields_information
        .iter()
        .map(|rx: &FieldReceiver| {
            let field_name = rx.ident.clone().unwrap();
            let field_type = rx.ty.clone();
            let transpose = rx.transpose.unwrap_or(receiver.transpose).transpose_read();

            let array_name = rx.rename.read_name_or_ident(&field_name);
            let is_attribute = rx.is_attribute;
            //let array_name = field_name.to_string();

            read::ReadInfo {field_name, field_type, transpose, array_name, is_attribute}

        }).collect();

    // build the writing body:
    let write_data: Vec<write::WriteInfo> = fields_information
        .iter()
        .map(|rx: &FieldReceiver| {
            let field_name = rx.ident.clone().unwrap();
            let field_type = rx.ty.clone();
            let transpose = rx.transpose.unwrap_or(receiver.transpose).transpose_write();
            let array_name = rx.rename.write_name_or_ident(&field_name);

            let mutate_on_write = rx.mutate_on_write.unwrap_or(receiver.mutate_on_write);
            let is_attribute = rx.is_attribute;

            write::WriteInfo {field_name, field_type, transpose, array_name, mutate_on_write, is_attribute}

        }).collect();


    let read_impl = read::read_codegen(receiver.ident.clone(), input.span(), &read_data)?;
    let write_impl = write::write_codegen(input.span(), &write_data)?;

    Ok(combine_impls(receiver.ident, receiver.generics, read_impl, write_impl).into())
}


fn combine_impls(ident: syn::Ident, generics: syn::Generics, read_body: proc_macro2::TokenStream, write_body: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let (imp, ty, wher) = generics.split_for_impl();

    quote::quote!(
        impl #imp hdf5_derive::ContainerIo for #ident #ty #wher {
            fn write_hdf5(&self, file: &hdf5_derive::File) -> Result<(), hdf5_derive::Error> {
                #write_body
            }
            fn read_hdf5(group: &hdf5_derive::Group) -> Result<Self, hdf5_derive::Error> 
                where Self: Sized
            {
                #read_body
            }
        }
    )
}
