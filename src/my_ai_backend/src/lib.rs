//! This module declares canister methods expected by the assets canister client.
use ic_cdk_macros::{export_candid, init, post_upgrade, pre_upgrade, query, update};
use itertools::Itertools;
use num_traits::ToPrimitive;
pub mod asset_certification;
pub mod evidence;
pub mod state_machine;
pub mod types;
mod url_decode;

pub use crate::state_machine::StableState;
use crate::{
    asset_certification::types::http::{
        CallbackFunc, HttpRequest, HttpResponse, StreamingCallbackHttpResponse,
        StreamingCallbackToken,
    },
    state_machine::{AssetDetails, CertifiedTree, EncodedAsset, State},
    types::*,
};
use asset_certification::types::{certification::AssetKey, rc_bytes::RcBytes};
use candid::Principal;
use ic_cdk::api::{call::ManualReply, caller, data_certificate, set_certified_data, time, trap};
use serde_bytes::ByteBuf;
use std::{cell::RefCell, ops::Deref, str::FromStr};

mod llm;
use anyhow::{Error, Result};
use candle::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
#[link_section = "icp:public supported_certificate_versions"]
pub static SUPPORTED_CERTIFICATE_VERSIONS: [u8; 3] = *b"1,2";

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

#[update]
fn set_model_content(arg: SetModelContentArguments) -> Result<SetModelContentResponse, String> {
    STATE.with(|s| s.borrow_mut().set_model_content(arg))
}

#[update]
fn load_config(arg: GetArg) {
    ic_cdk::println!("called load_config");
    let get_result = STATE.with(|s| s.borrow_mut().get_model_content(&arg.key));

    let config: Config = match get_result {
        Ok(content) => {
            ic_cdk::println!("content: {:?}", content);
            llm::CustomConfigTrait::from_slice(&content)
        }
        Err(e) => {
            ic_cdk::println!("error: {:?}", e);
            trap(&e.to_string());
        }
    };

    ic_cdk::println!("config: {:?}", config);
}
fn get_asset_content(arg: GetArg) -> Result<String, String> {
    let encoded_asset = get(arg.clone());
    let total_length = encoded_asset.total_length.0;
    let content_length = candid::Nat::from(encoded_asset.content.len()).0;
    let size_ceil = num_integer::div_ceil(total_length, content_length) - candid::Nat::from(1u32).0;
    let from_utf8_result = String::from_utf8_lossy(encoded_asset.content.as_ref());
    let mut content = from_utf8_result.to_string();
    for index in 0..size_ceil.to_usize().unwrap() {
        ic_cdk::println!("index: {:?}", index);
        let get_chunk_arg = GetChunkArg {
            key: arg.key.clone(),
            content_encoding: arg.accept_encodings[0].clone(),
            index: candid::Nat::from(index + 1),
            sha256: encoded_asset.sha256.clone(),
        };
        let chunk_response = get_chunk(get_chunk_arg);
        content += String::from_utf8_lossy(chunk_response.content.as_ref())
            .to_string()
            .as_str();
    }
    Ok(content)
}
#[update]
fn load_tokenizer(arg: GetArg) {
    ic_cdk::println!("call load_tokenizer");
    let get_result = STATE.with(|s| s.borrow_mut().get_model_content(&arg.key));
    let content = match get_result {
        Ok(content) => content,
        Err(e) => {
            ic_cdk::println!("error: {:?}", e);
            trap(&e.to_string());
        }
    };

    let tokenizer_result = Tokenizer::from_bytes(content);
    match tokenizer_result {
        Ok(tokenizer) => {
            ic_cdk::println!("loaded tokenizer.");
            ic_cdk::println!("tokenizer: {:?}", tokenizer)
        }
        Err(e) => {
            ic_cdk::println!("error: {:?}", e);
        }
    };
}

// #[query]
// fn load_model(arg: GetArg) {
//     ic_cdk::println!("call load_model");
//     let get_result = STATE.with(|s| s.borrow().get_model_content(arg.key));

//     match get_result {
//         Ok(weights) => {
//             ic_cdk::println!("Ok(content)");
//             let device = &Device::Cpu;
//             ic_cdk::println!("device: {:?}", device);
//             ic_cdk::println!("weights.len(): {:?}", weights.len());
//             let vb_result =
//                 VarBuilder::from_buffered_safetensors(weights.into_bytes(), DType::F32, device);
//             let vb = match vb_result {
//                 Ok(vb) => vb,
//                 Err(e) => {
//                     ic_cdk::println!("error: {:?}", e);
//                     trap(&e.to_string());
//                 }
//             };
//             ic_cdk::println!("loaded vb");
//             // let mixformer = MixFormer::new(&config, vb).unwrap();
//             // ic_cdk::println!("mixformer: {:?}", mixformer);
//             // let model = llm::SelectedModel::MixFormer(mixformer);
//             // ic_cdk::println!("model: {:?}", model);
//             // ic_cdk::api::print(format!("loaded the model"));
//             // let seed: u64 = 299792458;
//             // let logits_processor = LogitsProcessor::new(seed, None, None);
//             // ic_cdk::println!("loaded logits_processor.");
//             // let loaded_model = llm::Model {
//             //     model,
//             //     tokenizer,
//             //     logits_processor,
//             //     tokens: vec![],
//             //     repeat_penalty: 1.,
//             //     repeat_last_n: 64,
//             // };
//             // ic_cdk::println!("loaded_model: {:?}", loaded_model);
//         }
//         Err(e) => {
//             ic_cdk::println!("error: {:?}", e);
//         }
//     }
// }

#[update]
fn load_gguf(arg: GetArg) {
    ic_cdk::println!("call load_gguf");
    let vb_result = STATE.with(|s| {
        // let weights = model_content.content.clone();
        // ic_cdk::println!("weights.len(): {:?}", &model_content.content);
        candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
            &s.borrow()
                .model_store
                .get(&arg.key)
                .unwrap_or_else(|| trap(&format!("No model found for key {}", arg.key.to_string())))
                .content,
        )
    });

    // let weights = match get_result {
    //     Ok(chunks) => chunks,
    //     Err(e) => {
    //         ic_cdk::println!("error: {:?}", e);
    //         trap(&e.to_string());
    //     }
    // };

    // ic_cdk::println!("weights.len(): {:?}", weights.len());

    // let vb_result =
    //     candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(&weights);

    let vb = match vb_result {
        Ok(vb) => vb,
        Err(e) => {
            ic_cdk::println!("error: {:?}", e);
            trap(&e.to_string());
        }
    };

    ic_cdk::println!("loaded vb");
    let config_arg = GetArg {
        key: "/config.json".to_string(),
        accept_encodings: vec!["identity".to_string()],
    };
    let asset_config = get(config_arg);
    ic_cdk::println!("asset_config: {:?}", asset_config);

    let config: Config = llm::CustomConfigTrait::from_slice(&asset_config.content);

    ic_cdk::println!("config: {:?}", config);
    // let mixformer_result = QMixFormer::new(&config, vb);
    // ic_cdk::println!("mixformer_result: {:?}", mixformer_result);

    // let model = match mixformer_result {
    //     Ok(m) => llm::SelectedModel::Quantized(m),
    //     Err(e) => {
    //         ic_cdk::println!("error: {:?}", e);
    //         trap(&e.to_string());
    //     }
    // };

    // ic_cdk::println!("model: {:?}", model);
    // ic_cdk::api::print(format!("loaded the model"));
    // let seed: u64 = 299792458;
    // let logits_processor = LogitsProcessor::new(seed, None, None);
    // ic_cdk::println!("loaded logits_processor.");
    // let loaded_model = llm::Model {
    //     model,
    //     tokenizer,
    //     logits_processor,
    //     tokens: vec![],
    //     repeat_penalty: 1.,
    //     repeat_last_n: 64,
    // };
    // ic_cdk::println!("loaded_model: {:?}", loaded_model);
}

#[query]
fn api_version() -> u16 {
    1
}

#[update(guard = "is_manager_or_controller")]
fn authorize(other: Principal) {
    STATE.with(|s| s.borrow_mut().grant_permission(other, &Permission::Commit))
}

// #[update(guard = "is_manager_or_controller")]
#[update] // TODO: uncomment the above line and remove this one once the finish development
fn grant_permission(arg: GrantPermissionArguments) {
    STATE.with(|s| {
        s.borrow_mut()
            .grant_permission(arg.to_principal, &arg.permission)
    })
}

#[update]
async fn validate_grant_permission(arg: GrantPermissionArguments) -> Result<String, String> {
    Ok(format!(
        "grant {} permission to principal {}",
        arg.permission, arg.to_principal
    ))
}

#[update]
async fn deauthorize(other: Principal) {
    let check_access_result = if other == caller() {
        // this isn't "ManagePermissions" because these legacy methods only
        // deal with the Commit permission
        has_permission_or_is_controller(&Permission::Commit)
    } else {
        is_controller()
    };
    match check_access_result {
        Err(e) => trap(&e),
        Ok(_) => STATE.with(|s| s.borrow_mut().revoke_permission(other, &Permission::Commit)),
    }
}

#[update]
async fn revoke_permission(arg: RevokePermissionArguments) {
    let check_access_result = if arg.of_principal == caller() {
        has_permission_or_is_controller(&arg.permission)
    } else {
        has_permission_or_is_controller(&Permission::ManagePermissions)
    };
    match check_access_result {
        Err(e) => trap(&e),
        Ok(_) => STATE.with(|s| {
            s.borrow_mut()
                .revoke_permission(arg.of_principal, &arg.permission)
        }),
    }
}

#[update]
async fn validate_revoke_permission(arg: RevokePermissionArguments) -> Result<String, String> {
    Ok(format!(
        "revoke {} permission from principal {}",
        arg.permission, arg.of_principal
    ))
}

#[query(manual_reply = true)]
fn list_authorized() -> ManualReply<Vec<Principal>> {
    STATE.with(|s| ManualReply::one(s.borrow().list_permitted(&Permission::Commit)))
}

#[query(manual_reply = true)]
fn list_permitted(arg: ListPermittedArguments) -> ManualReply<Vec<Principal>> {
    STATE.with(|s| ManualReply::one(s.borrow().list_permitted(&arg.permission)))
}

#[update(guard = "is_controller")]
async fn take_ownership() {
    let caller = ic_cdk::api::caller();
    STATE.with(|s| s.borrow_mut().take_ownership(caller))
}

#[update]
async fn validate_take_ownership() -> Result<String, String> {
    Ok("revoke all permissions, then gives the caller Commit permissions".to_string())
}

#[query]
fn retrieve(key: AssetKey) -> RcBytes {
    STATE.with(|s| match s.borrow().retrieve(&key) {
        Ok(bytes) => bytes,
        Err(msg) => trap(&msg),
    })
}

#[update(guard = "can_commit")]
fn store(arg: StoreArg) {
    STATE.with(move |s| {
        if let Err(msg) = s.borrow_mut().store(arg, time()) {
            trap(&msg);
        }
        set_certified_data(&s.borrow().root_hash());
    });
}

#[update(guard = "can_prepare")]
fn create_batch() -> CreateBatchResponse {
    STATE.with(|s| match s.borrow_mut().create_batch(time()) {
        Ok(batch_id) => CreateBatchResponse { batch_id },
        Err(msg) => trap(&msg),
    })
}

#[update(guard = "can_prepare")]
fn create_chunk(arg: CreateChunkArg) -> CreateChunkResponse {
    STATE.with(|s| match s.borrow_mut().create_chunk(arg, time()) {
        Ok(chunk_id) => CreateChunkResponse { chunk_id },
        Err(msg) => trap(&msg),
    })
}

#[update(guard = "can_commit")]
fn create_asset(arg: CreateAssetArguments) {
    STATE.with(|s| {
        if let Err(msg) = s.borrow_mut().create_asset(arg) {
            trap(&msg);
        }
        set_certified_data(&s.borrow().root_hash());
    })
}

#[update(guard = "can_commit")]
fn set_asset_content(arg: SetAssetContentArguments) {
    STATE.with(|s| {
        if let Err(msg) = s.borrow_mut().set_asset_content(arg, time()) {
            trap(&msg);
        }
        set_certified_data(&s.borrow().root_hash());
    })
}

#[update(guard = "can_commit")]
fn unset_asset_content(arg: UnsetAssetContentArguments) {
    STATE.with(|s| {
        if let Err(msg) = s.borrow_mut().unset_asset_content(arg) {
            trap(&msg);
        }
        set_certified_data(&s.borrow().root_hash());
    })
}

#[update(guard = "can_commit")]
fn delete_asset(arg: DeleteAssetArguments) {
    STATE.with(|s| {
        s.borrow_mut().delete_asset(arg);
        set_certified_data(&s.borrow().root_hash());
    });
}

#[update(guard = "can_commit")]
fn clear() {
    STATE.with(|s| {
        s.borrow_mut().clear();
        set_certified_data(&s.borrow().root_hash());
    });
}

#[update(guard = "can_commit")]
fn commit_batch(arg: CommitBatchArguments) {
    STATE.with(|s| {
        if let Err(msg) = s.borrow_mut().commit_batch(arg, time()) {
            trap(&msg);
        }
        set_certified_data(&s.borrow().root_hash());
    });
}

#[update(guard = "can_prepare")]
fn propose_commit_batch(arg: CommitBatchArguments) {
    STATE.with(|s| {
        if let Err(msg) = s.borrow_mut().propose_commit_batch(arg) {
            trap(&msg);
        }
    });
}

#[update(guard = "can_prepare")]
fn compute_evidence(arg: ComputeEvidenceArguments) -> Option<ByteBuf> {
    STATE.with(|s| match s.borrow_mut().compute_evidence(arg) {
        Err(msg) => trap(&msg),
        Ok(maybe_evidence) => maybe_evidence,
    })
}

#[update(guard = "can_commit")]
fn commit_proposed_batch(arg: CommitProposedBatchArguments) {
    STATE.with(|s| {
        if let Err(msg) = s.borrow_mut().commit_proposed_batch(arg, time()) {
            trap(&msg);
        }
        set_certified_data(&s.borrow().root_hash());
    });
}

#[update]
fn validate_commit_proposed_batch(arg: CommitProposedBatchArguments) -> Result<String, String> {
    STATE.with(|s| s.borrow_mut().validate_commit_proposed_batch(arg))
}

#[update(guard = "can_prepare")]
fn delete_batch(arg: DeleteBatchArguments) {
    STATE.with(|s| {
        if let Err(msg) = s.borrow_mut().delete_batch(arg) {
            trap(&msg);
        }
    });
}

#[query]
fn get(arg: GetArg) -> EncodedAsset {
    STATE.with(|s| match s.borrow().get(arg) {
        Ok(asset) => asset,
        Err(msg) => trap(&msg),
    })
}

#[query]
fn get_chunk(arg: GetChunkArg) -> GetChunkResponse {
    STATE.with(|s| match s.borrow().get_chunk(arg) {
        Ok(content) => GetChunkResponse { content },
        Err(msg) => trap(&msg),
    })
}

#[query]
fn list() -> Vec<AssetDetails> {
    STATE.with(|s| s.borrow().list_assets())
}

#[query]
fn certified_tree() -> CertifiedTree {
    let certificate = data_certificate().unwrap_or_else(|| trap("no data certificate available"));

    STATE.with(|s| s.borrow().certified_tree(&certificate))
}

#[query]
fn http_request(req: HttpRequest) -> HttpResponse {
    let certificate = data_certificate().unwrap_or_else(|| trap("no data certificate available"));

    STATE.with(|s| {
        s.borrow().http_request(
            req,
            &certificate,
            CallbackFunc::new(ic_cdk::id(), "http_request_streaming_callback".to_string()),
        )
    })
}

#[query]
fn http_request_streaming_callback(token: StreamingCallbackToken) -> StreamingCallbackHttpResponse {
    STATE.with(|s| {
        s.borrow()
            .http_request_streaming_callback(token)
            .unwrap_or_else(|msg| trap(&msg))
    })
}

#[query]
fn get_asset_properties(key: AssetKey) -> AssetProperties {
    STATE.with(|s| {
        s.borrow()
            .get_asset_properties(key)
            .unwrap_or_else(|msg| trap(&msg))
    })
}

#[update(guard = "can_commit")]
fn set_asset_properties(arg: SetAssetPropertiesArguments) {
    STATE.with(|s| {
        if let Err(msg) = s.borrow_mut().set_asset_properties(arg) {
            trap(&msg);
        }
    })
}

#[update(guard = "can_prepare")]
fn get_configuration() -> ConfigurationResponse {
    STATE.with(|s| s.borrow().get_configuration())
}

#[update(guard = "can_commit")]
fn configure(arg: ConfigureArguments) {
    STATE.with(|s| s.borrow_mut().configure(arg))
}

#[update]
fn validate_configure(arg: ConfigureArguments) -> Result<String, String> {
    Ok(format!("configure: {:?}", arg))
}

fn can(permission: Permission) -> Result<(), String> {
    STATE.with(|s| {
        s.borrow()
            .can(&caller(), &permission)
            .then_some(())
            .ok_or_else(|| format!("Caller does not have {} permission", permission))
    })
}

fn can_commit() -> Result<(), String> {
    can(Permission::Commit)
}

fn can_prepare() -> Result<(), String> {
    can(Permission::Prepare)
}

fn has_permission_or_is_controller(permission: &Permission) -> Result<(), String> {
    let caller = caller();
    let has_permission = STATE.with(|s| s.borrow().has_permission(&caller, permission));
    let is_controller = ic_cdk::api::is_controller(&caller);
    if has_permission || is_controller {
        Ok(())
    } else {
        Err(format!(
            "Caller does not have {} permission and is not a controller.",
            permission
        ))
    }
}

fn is_manager_or_controller() -> Result<(), String> {
    has_permission_or_is_controller(&Permission::ManagePermissions)
}

fn is_controller() -> Result<(), String> {
    let caller = caller();
    if ic_cdk::api::is_controller(&caller) {
        Ok(())
    } else {
        Err("Caller is not a controller.".to_string())
    }
}

#[init]
fn init() {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        s.clear();
        s.grant_permission(caller(), &Permission::Commit);
    });
}

#[pre_upgrade]
fn pre_upgrade() {
    let stable_state: StableState = STATE.with(|s| s.take().into());
    ic_cdk::storage::stable_save((stable_state,)).expect("failed to save stable state");
}

#[post_upgrade]
fn post_upgrade() {
    let (stable_state,): (StableState,) =
        ic_cdk::storage::stable_restore().expect("failed to restore stable state");
    STATE.with(|s| {
        *s.borrow_mut() = State::from(stable_state);
        set_certified_data(&s.borrow().root_hash());
    });
}

export_candid!();
