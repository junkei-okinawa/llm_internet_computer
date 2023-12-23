use candid::{CandidType, Decode, Encode};
use ic_cdk_macros::{export_candid, post_upgrade, pre_upgrade, query, update};
use ic_stable_structures::storable::Bound;
use ic_stable_structures::{writer::Writer, Memory as _, StableBTreeMap, Storable};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::{
    borrow::Cow,
    cell::RefCell,
    cmp::{Eq, Ord, PartialEq, PartialOrd},
};
mod memory;
use memory::Memory;

// use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
// use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

// use candle::{DType, Device, Tensor};
// use candle_nn::{Activation, VarBuilder};
// use candle_transformers::generation::LogitsProcessor;
// use tokenizers::Tokenizer;

const MAX_SIZE_LLM: u32 = u32::MAX; // 4GB
const MAX_COUNT_PAT: u32 = 10;

#[derive(Eq, Ord, PartialEq, PartialOrd, CandidType, Deserialize, Debug, Clone)]
struct LLM {
    model_name: String,
    weight: Option<Vec<u8>>,
    tokenizer: Option<Vec<u8>>,
    config: Option<ConfigData>,
}

#[derive(Eq, Ord, PartialEq, PartialOrd, CandidType, Deserialize, Debug, Clone)]
struct ConfigData {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_inner: Option<usize>,
    pub n_head: usize,
    pub rotary_dim: usize,
    pub activation_function: String,
    pub layer_norm_epsilon: String,
    pub tie_word_embeddings: bool,
    pub pad_vocab_size_multiple: usize,
}

// #[derive(Deserialize)]
// struct CustomConfigStruct(Config);

// pub trait CustomConfigTrait {
//     fn from_file(file_config: ConfigData) -> Self;
// }

// impl CustomConfigTrait for Config {
//     fn from_file(data: ConfigData) -> Self {
//         let layer_norm_epsilon = data.layer_norm_epsilon.parse::<f64>().unwrap();
//         let activation_function = match data.activation_function.as_str() {
//             "gelu" => Activation::Gelu,
//             "new_gelu" => Activation::NewGelu,
//             "relu" => Activation::Relu,
//             "relu2" => Activation::Relu2,
//             "relu6" => Activation::Relu6,
//             "silu" => Activation::Silu,
//             "sigmoid" => Activation::Sigmoid,
//             "hard_sigmoid" => Activation::HardSigmoid,
//             "swiglu" => Activation::Swiglu,
//             "swish" => Activation::Swish,
//             "hard_swish" => Activation::HardSwish,
//             "elu" => Activation::Elu(layer_norm_epsilon),
//             "leaky_relu" => Activation::LeakyRelu(layer_norm_epsilon),
//             _ => Activation::NewGelu,
//         };

//         // data key contains "pad_vocab_size_multiple"?
//         let pad_vocab_size_multiple = match data.pad_vocab_size_multiple {
//             None => 64,
//             Some(pad_vocab_size_multiple) => pad_vocab_size_multiple,
//         };

//         Config::new(
//             data.vocab_size,
//             data.n_positions,
//             data.n_embd,
//             data.n_layer,
//             data.n_inner,
//             data.n_head,
//             data.rotary_dim,
//             activation_function,
//             layer_norm_epsilon,
//             data.tie_word_embeddings,
//             pad_vocab_size_multiple,
//         )
//     }
// }

#[update]
fn init_llm_config(model_name: String, config: ConfigData) -> Option<String> {
    let llm = match exist_llm(model_name.clone()) {
        true => {
            let llm_result = STATE.with(|s| s.borrow().stable_data.get(&model_name));
            let llm = match llm_result {
                Some(llm) => llm,
                None => return None,
            };
            StableBTreeMapLLM(LLM {
                model_name: llm.0.model_name.clone(),
                weight: llm.0.weight.clone(),
                tokenizer: llm.0.tokenizer.clone(),
                config: Some(config),
            })
        }
        false => StableBTreeMapLLM(LLM {
            model_name: model_name.clone(),
            weight: None,
            tokenizer: None,
            config: Some(config),
        }),
    };
    insert_llm(model_name.clone(), llm);
    Some(model_name)
}

#[derive(CandidType, Ord, PartialOrd, Eq, PartialEq, Debug, Clone, Deserialize)]
struct StableBTreeMapLLM(LLM);

impl Storable for StableBTreeMapLLM {
    fn to_bytes(&self) -> std::borrow::Cow<[u8]> {
        Cow::Owned(Encode!(self).unwrap())
    }
    fn from_bytes(bytes: Cow<'_, [u8]>) -> Self {
        Decode!(&bytes, Self).unwrap()
    }
    const BOUND: Bound = Bound::Bounded {
        max_size: MAX_SIZE_LLM,
        is_fixed_size: false,
    };
}

// #[derive(Serialize, Deserialize, Clone, CandidType)]
// struct TmpData {
//     data: BTreeMap<String, TmpChunks>, // key: model_name & "_" & content_type
// }

#[derive(Serialize, Deserialize, Clone, CandidType)]
struct TmpChunks {
    chunks: HashMap<u32, Vec<u8>>, // key: chunk_id & value: chunk
    data_name: String,
    content_type: ContentType,
}

#[derive(Serialize, Deserialize, Clone, CandidType)]
enum ContentType {
    Weight,
    Tokenizer,
}

#[derive(Serialize, Deserialize, Clone, CandidType)]
pub enum CountPat {
    Pat0,
    Pat1,
    Pat2,
    Pat3,
    Pat4,
    Pat5,
    Pat6,
    Pat7,
    Pat8,
    Pat9,
}

// The state of the canister.
#[derive(Serialize, Deserialize)]
struct State {
    // Data that lives on the heap.
    // This is an example for data that would need to be serialized/deserialized
    // on every upgrade for it to be persisted.
    data_on_the_heap: Vec<u8>,

    // An example `StableBTreeMap`. Data stored in `StableBTreeMap` doesn't need to
    // be serialized/deserialized in upgrades, so we tell serde to skip it.
    #[serde(skip, default = "init_stable_data")]
    stable_data: StableBTreeMap<String, StableBTreeMapLLM, Memory>,
    tmp_data0: BTreeMap<String, TmpChunks>,
    tmp_data1: BTreeMap<String, TmpChunks>,
    tmp_data2: BTreeMap<String, TmpChunks>,
    tmp_data3: BTreeMap<String, TmpChunks>,
    tmp_data4: BTreeMap<String, TmpChunks>,
    tmp_data5: BTreeMap<String, TmpChunks>,
    tmp_data6: BTreeMap<String, TmpChunks>,
    tmp_data7: BTreeMap<String, TmpChunks>,
    tmp_data8: BTreeMap<String, TmpChunks>,
    tmp_data9: BTreeMap<String, TmpChunks>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            data_on_the_heap: vec![],
            stable_data: init_stable_data(),
            tmp_data0: BTreeMap::new(),
            tmp_data1: BTreeMap::new(),
            tmp_data2: BTreeMap::new(),
            tmp_data3: BTreeMap::new(),
            tmp_data4: BTreeMap::new(),
            tmp_data5: BTreeMap::new(),
            tmp_data6: BTreeMap::new(),
            tmp_data7: BTreeMap::new(),
            tmp_data8: BTreeMap::new(),
            tmp_data9: BTreeMap::new(),
        }
    }
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn get_count_pat(id: u32) -> CountPat {
    match id % 10 {
        0 => CountPat::Pat0,
        1 => CountPat::Pat1,
        2 => CountPat::Pat2,
        3 => CountPat::Pat3,
        4 => CountPat::Pat4,
        5 => CountPat::Pat5,
        6 => CountPat::Pat6,
        7 => CountPat::Pat7,
        8 => CountPat::Pat8,
        9 => CountPat::Pat9,
        _ => panic!("Invalid memory id"),
    }
}

fn exist_tmp_data(count_pat: CountPat, model_name: String, content_type: String) -> bool {
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow()
                .tmp_data0
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow()
                .tmp_data1
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat2 => STATE.with(|s| {
            s.borrow()
                .tmp_data2
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat3 => STATE.with(|s| {
            s.borrow()
                .tmp_data3
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat4 => STATE.with(|s| {
            s.borrow()
                .tmp_data4
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat5 => STATE.with(|s| {
            s.borrow()
                .tmp_data5
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat6 => STATE.with(|s| {
            s.borrow()
                .tmp_data6
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat7 => STATE.with(|s| {
            s.borrow()
                .tmp_data7
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat8 => STATE.with(|s| {
            s.borrow()
                .tmp_data8
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat9 => STATE.with(|s| {
            s.borrow()
                .tmp_data9
                .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
        }),
    }
}

fn exist_temp_chunk(
    count_pat: CountPat,
    model_name: String,
    content_type: String,
    chunk_id: u32,
) -> bool {
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow()
                .tmp_data0
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow()
                .tmp_data1
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat2 => STATE.with(|s| {
            s.borrow()
                .tmp_data2
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat3 => STATE.with(|s| {
            s.borrow()
                .tmp_data3
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat4 => STATE.with(|s| {
            s.borrow()
                .tmp_data4
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat5 => STATE.with(|s| {
            s.borrow()
                .tmp_data5
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat6 => STATE.with(|s| {
            s.borrow()
                .tmp_data6
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat7 => STATE.with(|s| {
            s.borrow()
                .tmp_data7
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat8 => STATE.with(|s| {
            s.borrow()
                .tmp_data8
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
        CountPat::Pat9 => STATE.with(|s| {
            s.borrow()
                .tmp_data9
                .get(&(model_name.clone() + "_" + &content_type.clone()))
                .unwrap()
                .chunks
                .contains_key(&chunk_id)
        }),
    }
}

fn insert_tmp_chunks(
    count_pat: CountPat,
    model_name: String,
    content_type: String,
    tmp_chunks: TmpChunks,
) {
    let key = model_name.clone() + "_" + &content_type.clone();
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| s.borrow_mut().tmp_data0.insert(key, tmp_chunks)),
        CountPat::Pat1 => STATE.with(|s| s.borrow_mut().tmp_data1.insert(key, tmp_chunks)),
        CountPat::Pat2 => STATE.with(|s| s.borrow_mut().tmp_data2.insert(key, tmp_chunks)),
        CountPat::Pat3 => STATE.with(|s| s.borrow_mut().tmp_data3.insert(key, tmp_chunks)),
        CountPat::Pat4 => STATE.with(|s| s.borrow_mut().tmp_data4.insert(key, tmp_chunks)),
        CountPat::Pat5 => STATE.with(|s| s.borrow_mut().tmp_data5.insert(key, tmp_chunks)),
        CountPat::Pat6 => STATE.with(|s| s.borrow_mut().tmp_data6.insert(key, tmp_chunks)),
        CountPat::Pat7 => STATE.with(|s| s.borrow_mut().tmp_data7.insert(key, tmp_chunks)),
        CountPat::Pat8 => STATE.with(|s| s.borrow_mut().tmp_data8.insert(key, tmp_chunks)),
        CountPat::Pat9 => STATE.with(|s| s.borrow_mut().tmp_data9.insert(key, tmp_chunks)),
    };
}

fn update_chunks(
    count_pat: CountPat,
    model_name: String,
    content_type: String,
    chunk_id: u32,
    chunk: Vec<u8>,
) {
    let key = model_name.clone() + "_" + &content_type.clone();
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data0
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data1
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat2 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data2
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat3 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data3
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat4 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data4
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat5 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data5
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat6 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data6
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat7 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data7
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat8 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data8
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
        CountPat::Pat9 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data9
                .get_mut(&key)
                .unwrap()
                .chunks
                .insert(chunk_id, chunk.clone())
        }),
    };
}

fn get_tmp_chunk(
    count_pat: CountPat,
    model_name: String,
    content_type: String,
    chunk_id: u32,
) -> Vec<u8> {
    let key = model_name.clone() + "_" + &content_type.clone();
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow()
                .tmp_data0
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow()
                .tmp_data1
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat2 => STATE.with(|s| {
            s.borrow()
                .tmp_data2
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat3 => STATE.with(|s| {
            s.borrow()
                .tmp_data3
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat4 => STATE.with(|s| {
            s.borrow()
                .tmp_data4
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat5 => STATE.with(|s| {
            s.borrow()
                .tmp_data5
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat6 => STATE.with(|s| {
            s.borrow()
                .tmp_data6
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat7 => STATE.with(|s| {
            s.borrow()
                .tmp_data7
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat8 => STATE.with(|s| {
            s.borrow()
                .tmp_data8
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
        CountPat::Pat9 => STATE.with(|s| {
            s.borrow()
                .tmp_data9
                .get(&key)
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone()
        }),
    }
}

#[update]
pub async fn create_chunk(
    model_name: String,
    content_type: String,
    chunk: Vec<u8>,
    counter: u32,
) -> Result<u32, String> {
    ic_cdk::println!("Chunk Len: {:?}", chunk.len());
    let count_pat = get_count_pat(counter);

    if !exist_tmp_data(count_pat.clone(), model_name.clone(), content_type.clone()) {
        let mut tmp_chunks = TmpChunks {
            chunks: HashMap::new(),
            data_name: model_name.clone(),
            content_type: match content_type.as_str() {
                "weight" => ContentType::Weight,
                "tokenizer" => ContentType::Tokenizer,
                _ => return Err("Data type is not set".to_string()),
            },
        };
        tmp_chunks.chunks.insert(counter, chunk);
        insert_tmp_chunks(
            count_pat,
            model_name.clone(),
            content_type.clone(),
            tmp_chunks,
        );
    } else {
        update_chunks(count_pat, model_name, content_type, counter, chunk);
    }
    ic_cdk::println!("Chunk ID: {:?}", counter);
    Ok(counter)
}

#[update]
pub async fn commit_batch(
    model_name: String,
    content_type: String,
    mut chunk_ids: Vec<u32>,
) -> Result<String, String> {
    ic_cdk::println!("called commit_batch. Chunk IDs: {:?}", chunk_ids);

    if exist_llm(model_name.clone()) {
        ic_cdk::println!("exist llm. model_name: {:?}", model_name);
        ic_cdk::println!(
            "init llm content. model_name: {:?}, content_type: {:?}",
            model_name,
            content_type
        );
        let _ = match content_type.as_str() {
            "weight" => update_llm_weight(model_name.clone(), Some(vec![])),
            "tokenizer" => update_llm_tokenizer(model_name.clone(), Some(vec![])),
            _ => return Err("Data type is not set".to_string()),
        };
    } else {
        ic_cdk::println!("create llm. content_type: {:?}", content_type);
        let llm = match content_type.as_str() {
            "weight" => StableBTreeMapLLM(LLM {
                model_name: model_name.clone(),
                weight: Some(vec![]),
                tokenizer: None,
                config: None,
            }),
            "tokenizer" => StableBTreeMapLLM(LLM {
                model_name: model_name.clone(),
                weight: None,
                tokenizer: Some(vec![]),
                config: None,
            }),
            _ => return Err("Data type is not set".to_string()),
        };
        insert_llm(model_name.clone(), llm);
    };

    ic_cdk::println!("start!!! update llm content.");
    chunk_ids.sort();
    let last = chunk_ids.last().unwrap();
    let mut chunks = vec![];
    for chunk_id in chunk_ids.clone() {
        let count_pat = get_count_pat(chunk_id.clone());

        if !exist_tmp_data(
            get_count_pat(chunk_id),
            model_name.clone(),
            content_type.clone(),
        ) {
            return Err("Chunk id does not exist".to_string());
        }

        if !exist_temp_chunk(
            count_pat.clone(),
            model_name.clone(),
            content_type.clone(),
            chunk_id,
        ) {
            return Err("Chunk id does not exist".to_string());
        }

        let chunk = get_tmp_chunk(
            count_pat.clone(),
            model_name.clone(),
            content_type.clone(),
            chunk_id,
        );

        ic_cdk::println!("geted tmp chunk. chunk id: {:?}", chunk_id);

        chunks.extend(chunk);
        ic_cdk::println!("chunks.len(): {:?}", chunks.len());
        if chunk_id % 10 == 0 || chunk_id == *last {
            // chunks は 可変参照渡しとし、メモリリークを防ぐ
            let extend_result = match content_type.as_str() {
                "weight" => extend_llm_weight(model_name.clone(), chunks),
                "tokenizer" => extend_llm_tokenizer(model_name.clone(), chunks),
                _ => return Err("Data type is not set".to_string()),
            };
            chunks = match extend_result {
                Ok(chunks) => chunks,
                Err(err) => return Err(err),
            };
            ic_cdk::println!("extended chunks. chunk id: {:?}", chunk_id);
        }
    }

    ic_cdk::println!(
        "!!! updated. iim content. content_type {} !!!",
        content_type.clone()
    );

    let (removed_content, result_vec) = remove_tmp_data(model_name.clone(), content_type.clone());
    ic_cdk::println!(
        "!!! removed tmp_data. {:?} result_vec: {:?}!!!",
        removed_content,
        result_vec
    );

    ic_cdk::println!("!!! finished. commit_batch !!!");

    Ok(model_name + "_" + &content_type)
}

#[query]
fn exist_llm(key: String) -> bool {
    STATE.with(|s| s.borrow().stable_data.contains_key(&key))
}

#[query]
fn exist_llm_content(model_name: String, content_type: String) -> bool {
    let exist_llm = exist_llm(model_name.clone());
    if !exist_llm {
        return false;
    };
    STATE.with(|s| match content_type.as_str() {
        "weight" => match s.borrow().stable_data.get(&model_name).unwrap().0.weight {
            Some(_) => true,
            None => false,
        },
        "tokenizer" => match s.borrow().stable_data.get(&model_name).unwrap().0.tokenizer {
            Some(_) => true,
            None => false,
        },
        _ => false,
    })
}

// Retrieves the value associated with the given key in the stable data if it exists.
#[query]
fn get_llm_keys() -> Vec<String> {
    STATE.with(|s| {
        s.borrow()
            .stable_data
            .iter()
            .map(|(k, _)| k.clone())
            .collect()
    })
}

// Inserts an entry into the map and returns the previous value of the key from stable data
// if it exists.
#[update]
fn insert_llm(key: String, value: StableBTreeMapLLM) -> Option<String> {
    STATE.with(|s| s.borrow_mut().stable_data.insert(key.clone(), value));
    Some(key)
}

#[update]
fn update_llm_config(key: String, config: ConfigData) -> Option<String> {
    let llm_result = STATE.with(|s| s.borrow().stable_data.get(&key));
    let mut llm = match llm_result {
        Some(llm) => llm,
        None => return None,
    };
    llm = StableBTreeMapLLM(LLM {
        model_name: llm.0.model_name.clone(),
        weight: llm.0.weight.clone(),
        tokenizer: llm.0.tokenizer.clone(),
        config: Some(config),
    });
    STATE.with(|s| s.borrow_mut().stable_data.insert(key.clone(), llm));
    Some(key)
}

#[update]
fn update_llm_weight(key: String, weight: Option<Vec<u8>>) -> Option<String> {
    let llm_result = STATE.with(|s| s.borrow().stable_data.get(&key));
    let mut llm = match llm_result {
        Some(llm) => llm,
        None => return None,
    };
    llm = StableBTreeMapLLM(LLM {
        model_name: llm.0.model_name.clone(),
        weight: weight,
        tokenizer: llm.0.tokenizer.clone(),
        config: llm.0.config.clone(),
    });
    STATE.with(|s| s.borrow_mut().stable_data.insert(key.clone(), llm));
    Some(key)
}

#[update]
fn update_llm_tokenizer(key: String, tokenizer: Option<Vec<u8>>) -> Option<String> {
    let llm_result = STATE.with(|s| s.borrow().stable_data.get(&key));
    let mut llm = match llm_result {
        Some(llm) => llm,
        None => return None,
    };
    llm = StableBTreeMapLLM(LLM {
        model_name: llm.0.model_name.clone(),
        weight: llm.0.weight.clone(),
        tokenizer: tokenizer,
        config: llm.0.config.clone(),
    });
    STATE.with(|s| s.borrow_mut().stable_data.insert(key.clone(), llm));
    Some(key)
}

#[update]
fn extend_llm_weight(key: String, mut chunks: Vec<u8>) -> Result<Vec<u8>, String> {
    ic_cdk::println!("start extend_llm_weight.");
    let exist_llm = STATE.with(|s| s.borrow().stable_data.contains_key(&key));
    if !exist_llm {
        return Err("llm Not Found.".to_string());
    };

    ic_cdk::println!("exist_llm: {:?}", exist_llm);
    let option_weight = STATE.with(|s| s.borrow().stable_data.get(&key).unwrap().0.weight);
    ic_cdk::println!("option_weight.is_none(): {:?}", option_weight.is_none());
    match option_weight {
        Some(mut weight) => {
            ic_cdk::println!("option_weight is some.");
            weight.extend(chunks);
            ic_cdk::println!("extended weight.len(): {:?}", weight.len());
            STATE.with(|s| {
                let llm = s.borrow_mut().stable_data.get(&key).unwrap().0;
                STATE.with(|s| {
                    s.borrow_mut().stable_data.insert(
                        key.clone(),
                        StableBTreeMapLLM(LLM {
                            model_name: llm.model_name,
                            weight: Some(weight),
                            tokenizer: llm.tokenizer,
                            config: llm.config,
                        }),
                    )
                });
            });
            ic_cdk::println!("saved extended weight!!!");
        }
        None => return Err("weight Not Found.".to_string()),
    };
    chunks = vec![];
    Ok(chunks)
}

#[update]
fn extend_llm_tokenizer(key: String, mut chunks: Vec<u8>) -> Result<Vec<u8>, String> {
    let exist_llm = STATE.with(|s| s.borrow().stable_data.contains_key(&key));
    if !exist_llm {
        return Err("llm Not Found.".to_string());
    };
    let option_tokenizer = STATE.with(|s| s.borrow().stable_data.get(&key).unwrap().0.tokenizer);
    match option_tokenizer {
        Some(mut tokenizer) => {
            tokenizer.extend(chunks);
            STATE.with(|s| {
                let llm = s.borrow_mut().stable_data.get(&key).unwrap().0;
                STATE.with(|s| {
                    s.borrow_mut().stable_data.insert(
                        key.clone(),
                        StableBTreeMapLLM(LLM {
                            model_name: llm.model_name,
                            weight: llm.weight,
                            tokenizer: Some(tokenizer),
                            config: llm.config,
                        }),
                    )
                });
            });
        }
        None => return Err("tokenizer Not Found.".to_string()),
    };
    chunks = vec![];
    Ok(chunks)
}

#[query]
fn get_llm_ditaile(
    key: String,
) -> Option<(String, Option<usize>, Option<usize>, Option<ConfigData>)> {
    let llm_result = STATE.with(|s| s.borrow().stable_data.get(&key));
    match llm_result {
        Some(llm) => {
            let weight_len = match &llm.0.weight {
                Some(weight) => {
                    ic_cdk::api::print(format!("weight: {:?}", weight));
                    Some(weight.len())
                }
                None => None,
            };
            let tokenizer_len = match &llm.0.tokenizer {
                Some(tokenizer) => {
                    ic_cdk::api::print(format!("tokenizer: {:?}", tokenizer));
                    Some(tokenizer.len())
                }
                None => None,
            };

            ic_cdk::api::print(format!("model_name: {:?}", llm.0.model_name));
            ic_cdk::api::print(format!("weight_len: {:?}", weight_len));
            ic_cdk::api::print(format!("tokenizer_len: {:?}", tokenizer_len));
            ic_cdk::api::print(format!("config: {:?}", llm.0.config));

            Some((llm.0.model_name, weight_len, tokenizer_len, llm.0.config))
        }
        None => None,
    }
}

fn remove_tmp_content(count_pat: CountPat, model_name: String, content_type: String) -> bool {
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data0
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data1
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat2 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data2
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat3 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data3
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat4 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data4
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat5 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data5
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat6 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data6
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat7 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data7
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat8 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data8
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
        CountPat::Pat9 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data9
                .remove(&(model_name.clone() + "_" + &content_type.clone()))
        }),
    };
    true
}

#[update]
pub fn remove_tmp_data(model_name: String, content_type: String) -> (String, Vec<bool>) {
    let mut result = vec![];
    for i in 0..MAX_COUNT_PAT {
        let count_pat = get_count_pat(i);
        let exist_tmp_data =
            exist_tmp_data(count_pat.clone(), model_name.clone(), content_type.clone());
        if !exist_tmp_data {
            result.push(false);
            continue;
        }
        result.push(remove_tmp_content(
            count_pat.clone(),
            model_name.clone(),
            content_type.clone(),
        ));
    }
    (
        format!(
            "model_name: {:?}, content_type: {:?}",
            model_name, content_type
        ),
        result,
    )
}

// Sets the data that lives on the heap.
#[update]
fn set_heap_data(data: Vec<u8>) {
    STATE.with(|s| s.borrow_mut().data_on_the_heap = data);
}

// Retrieves the data that lives on the heap.
#[query]
fn get_heap_data() -> Vec<u8> {
    STATE.with(|s| s.borrow().data_on_the_heap.clone())
}

// A pre-upgrade hook for serializing the data stored on the heap.
#[pre_upgrade]
fn pre_upgrade() {
    // Serialize the state.
    // This example is using CBOR, but you can use any data format you like.
    let mut state_bytes = vec![];
    STATE
        .with(|s| ciborium::ser::into_writer(&*s.borrow(), &mut state_bytes))
        .expect("failed to encode state");

    // Write the length of the serialized bytes to memory, followed by the
    // by the bytes themselves.
    let len = state_bytes.len() as u32;
    let mut memory = memory::get_upgrades_memory();
    let mut writer = Writer::new(&mut memory, 0);
    writer.write(&len.to_le_bytes()).unwrap();
    writer.write(&state_bytes).unwrap()
}

// A post-upgrade hook for deserializing the data back into the heap.
#[post_upgrade]
fn post_upgrade() {
    let memory = memory::get_upgrades_memory();

    // Read the length of the state bytes.
    let mut state_len_bytes = [0; 4];
    memory.read(0, &mut state_len_bytes);
    let state_len = u32::from_le_bytes(state_len_bytes) as usize;

    // Read the bytes
    let mut state_bytes = vec![0; state_len];
    memory.read(4, &mut state_bytes);

    // Deserialize and set the state.
    let state = ciborium::de::from_reader(&*state_bytes).expect("failed to decode state");
    STATE.with(|s| *s.borrow_mut() = state);
}

fn init_stable_data() -> StableBTreeMap<String, StableBTreeMapLLM, Memory> {
    StableBTreeMap::init(crate::memory::get_stable_btree_memory())
}

#[query]
fn greet(name: String) -> String {
    format!("Hello, {}!", name)
}

export_candid!();
