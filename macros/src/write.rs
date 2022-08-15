use proc_macro2::TokenStream;
use syn::Result;
use quote::quote;
use proc_macro2::Span;

pub(crate) struct WriteInfo {
    pub(crate) field_name: syn::Ident,
    pub(crate) field_type: syn::Type,
    pub(crate) array_name: String,
    pub(crate) transpose: bool,
    pub(crate) mutate_on_write: bool,
    pub(crate) is_attribute: bool,
}

pub(crate) fn write_codegen(span: Span, arrays: &[WriteInfo]) -> Result<TokenStream> {
    let mut body = quote!();

    for array_or_attribute in arrays {
        body = if array_or_attribute.is_attribute {
            attribute_codegen(body, array_or_attribute, span)
        } else {
            array_codegen(body, array_or_attribute, span)
        }
    }

    // generate the full method implementation
    let full_impl = quote!(
        use ndarray::ShapeBuilder;
        #body

        Ok(())
    );

    Ok(full_impl)
}

fn array_codegen(mut body: TokenStream, array: &WriteInfo, span: Span) -> TokenStream {
    let WriteInfo {field_name, field_type, array_name, transpose, mutate_on_write, is_attribute: _} = array;

    let array_name_literal = syn::LitStr::new(array_name, span);

    // transpose the array if we need to 
    let (target_arr, target_arr_decl) = if *transpose {
        let ident = syn::Ident::new(&format!("tmp_{}", field_name), field_name.span());

        let target = quote!(#ident);
        // we have to make a copy of the array in order for us to guarantee the correct layout
        let target_arr_decl = quote!(
            let mut #target = ndarray::Array::zeros(self.#field_name.t().dim()); 
            #target.assign(&self.#field_name.t());
        );


        (target , target_arr_decl)
    } else {
        (quote!(self.#field_name), quote!())
    };


    // if we are mutating, we can avoid creating the dataset and instead simply fetch an
    // existing dataset
    let fetch_dataset = if *mutate_on_write {
        quote!(
            let #field_name = file.dataset(#array_name_literal)
                .map_err(|e| hdf5_derive::FetchDataset::from_field_name(#array_name_literal, e))?;
        )
    } else {
        quote!(
            let #field_name = file.new_dataset::<<#field_type as hdf5_derive::ArrayType>::Ty>()
                .shape(#target_arr.shape())
                .create(#array_name_literal)
                .map_err(|e| hdf5_derive::CreateDataset::from_field_name(#array_name_literal, e))?;
        )
    };

    // TODO: give some more error handling on this thing here
    // so the user knows what dataset was missing
    body = quote!(
        #body

        #target_arr_decl

        // fetch_dataset provides the declaration for #field_name based on if 
        // we are mutating the array or not
        #fetch_dataset

        #field_name.write(&#target_arr)
            .map_err(|e| hdf5_derive::WriteArray::from_field_name(#array_name_literal, e))?;
    );

    body
}

fn attribute_codegen(mut body: TokenStream, array: &WriteInfo, span: Span) -> TokenStream {
    let WriteInfo {field_name, field_type, array_name, transpose: _, mutate_on_write, is_attribute: _} = array;

    let attribute_name_literal = syn::LitStr::new(array_name, span);

    let target_arr = quote!(self.#field_name);

    // if we are mutating, we can avoid creating the attribute and instead simply fetch an
    // existing dataset
    let fetch_dataset = if *mutate_on_write {
        quote!(
            let #field_name = file.attr(#attribute_name_literal)
                .map_err(|e| hdf5_derive::FetchAttribute::from_field_name(#attribute_name_literal, e))?;
        )
    } else {
        quote!(
            let #field_name = file.new_attr::<#field_type>()
                .create(#attribute_name_literal)
                .map_err(|e| hdf5_derive::CreateAttribute::from_field_name(#attribute_name_literal, e))?;
        )
    };

    // TODO: give some more error handling on this thing here
    // so the user knows what dataset was missing
    body = quote!(
        #body

        // fetch_dataset provides the declaration for #field_name based on if 
        // we are mutating the array or not
        #fetch_dataset

        #field_name.write_scalar(&#target_arr)
            .map_err(|e| hdf5_derive::WriteAttribute::from_field_name(#attribute_name_literal, e))?;
    );

    body
}
