//use proc_macro::TokenStream;
use proc_macro2::TokenStream;
use syn::Result;
use quote::quote;
use proc_macro2::Span;
use syn::punctuated::Punctuated;

pub(crate) struct ReadInfo {
    pub(crate) field_name: syn::Ident,
    pub(crate) field_type: syn::Type,
    pub(crate) array_name: String,
    pub(crate) transpose: bool,
    pub(crate) is_attribute: bool,
}

pub(crate) fn read_codegen(ident: syn::Ident, span: Span, arrays: &[ReadInfo]) -> Result<TokenStream> {
    let mut body = quote!();

    for array_or_attribute in arrays {
        body = if array_or_attribute.is_attribute {
            attribute_codegen(body, &array_or_attribute, span)
        } else {
            array_codegen(body, &array_or_attribute, span)
        }
    }

    // build the final return statement
    let punct : Punctuated<syn::Ident, syn::Token![,]> = arrays.iter().map(|arr| arr.field_name.clone()).collect();
    let return_statement = quote!(Ok(#ident { #punct }));

    // generate the full method implementation
    let full_impl = quote!(
        #body

        #return_statement
    );

    Ok(full_impl)
}

fn array_codegen(mut body: TokenStream, info: &ReadInfo, span: Span) -> TokenStream {
    let ReadInfo {field_name, field_type, array_name, transpose, is_attribute: _} = info;

    let array_name_literal = syn::LitStr::new(&array_name, span);

    body = quote!(
        #body

        let #field_name = group.dataset(#array_name_literal)
            .map_err(|e| hdf5_derive::MissingDataset::from_field_name(#array_name_literal, e))?;
        let #field_name : #field_type = #field_name.read()
            .map_err(|e| hdf5_derive::SerializeArray::from_field_name(#array_name_literal, e))?;
    );

    // transpose the array if we need to 
    if *transpose {
        body = quote!(
            #body
            let #field_name = #field_name.reversed_axes();
        )
    }

    body
}

fn attribute_codegen(mut body: TokenStream, info: &ReadInfo, span: Span) -> TokenStream {
    let ReadInfo {field_name, field_type, array_name, transpose: _, is_attribute: _} = info;

    let attribute_name_literal = syn::LitStr::new(&array_name, span);

    body = quote!(
        #body

        let #field_name = group.attr(#attribute_name_literal)
            .map_err(|e| hdf5_derive::MissingAttribute::from_field_name(#attribute_name_literal, e))?;
        let #field_name : #field_type = #field_name.read_scalar()
            .map_err(|e| hdf5_derive::SerializeAttribute::from_field_name(#attribute_name_literal, e))?;
    );

    body
}
