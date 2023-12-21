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

const MAX_SIZE_LLM: u32 = u32::MAX; // 4GB
const MAX_COUNT: u32 = 2;

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
    pub pad_vocab_size_multiple: Option<usize>,
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
    // Pat2,
    // Pat3,
    // Pat4,
    // Pat5,
    // Pat6,
    // Pat7,
    // Pat8,
    // Pat9,
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
    // tmp_data2: BtreeMap<String, TmpChunks>,
    // tmp_data3: BtreeMap<String, TmpChunks>,
    // tmp_data4: BtreeMap<String, TmpChunks>,
    // tmp_data5: BtreeMap<String, TmpChunks>,
    // tmp_data6: BtreeMap<String, TmpChunks>,
    // tmp_data7: BtreeMap<String, TmpChunks>,
    // tmp_data8: BtreeMap<String, TmpChunks>,
    // tmp_data9: BtreeMap<String, TmpChunks>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            data_on_the_heap: vec![],
            stable_data: init_stable_data(),
            tmp_data0: BTreeMap::new(),
            tmp_data1: BTreeMap::new(),
            // tmp_data2: BTreeMap::new(),
            // tmp_data3: BTreeMap::new(),
            // tmp_data4: BTreeMap::new(),
            // tmp_data5: BTreeMap::new(),
            // tmp_data6: BTreeMap::new(),
            // tmp_data7: BTreeMap::new(),
            // tmp_data8: BTreeMap::new(),
            // tmp_data9: BTreeMap::new(),
        }
    }
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn get_count_pat(counter: u32) -> Result<CountPat, String> {
    match counter % MAX_COUNT {
        0 => Ok(CountPat::Pat0),
        1 => Ok(CountPat::Pat1),
        // 2 => Ok(CountPat::Pat2),
        // 3 => Ok(CountPat::Pat3),
        // 4 => Ok(CountPat::Pat4),
        // 5 => Ok(CountPat::Pat5),
        // 6 => Ok(CountPat::Pat6),
        // 7 => Ok(CountPat::Pat7),
        // 8 => Ok(CountPat::Pat8),
        // 9 => Ok(CountPat::Pat9),
        _ => Err("Counter is not set".to_string()),
    }
}

fn exist_tmp_data(count_pat: CountPat, model_name: String, content_type: String) -> bool {
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow()
                .tmp_data0
                .contains_key(&(model_name.clone() + "_" + &content_type))
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow()
                .tmp_data1
                .contains_key(&(model_name.clone() + "_" + &content_type))
        }),
        // CountPat::Pat2 => STATE.with(|s| {
        //     s.borrow()
        //         .tmp_data2
        //         .contains_key(&(model_name.clone() + "_" + &content_type))
        // }),
        // CountPat::Pat3 => STATE.with(|s| {
        //     s.borrow()
        //         .tmp_data3
        //         .contains_key(&(model_name.clone() + "_" + &content_type))
        // }),
        // CountPat::Pat4 => STATE.with(|s| {
        //     s.borrow()
        //         .tmp_data4
        //         .contains_key(&(model_name.clone() + "_" + &content_type))
        // }),
        // CountPat::Pat5 => STATE.with(|s| {
        //     s.borrow()
        //         .tmp_data5
        //         .contains_key(&(model_name.clone() + "_" + &content_type))
        // }),
        // CountPat::Pat6 => STATE.with(|s| {
        //     s.borrow()
        //         .tmp_data6
        //         .contains_key(&(model_name.clone() + "_" + &content_type))
        // }),
        // CountPat::Pat7 => STATE.with(|s| {
        //     s.borrow()
        //         .tmp_data7
        //         .contains_key(&(model_name.clone() + "_" + &content_type))
        // }),
        // CountPat::Pat8 => STATE.with(|s| {
        //     s.borrow()
        //         .tmp_data8
        //         .contains_key(&(model_name.clone() + "_" + &content_type))
        // }),
        // CountPat::Pat9 => STATE.with(|s| {
        //     s.borrow()
        //         .tmp_data9
        //         .contains_key(&(model_name.clone() + "_" + &content_type))
        // }),
    }
}

#[update]
pub async fn create_chunk(
    model_name: String,
    content_type: String,
    chunk: Vec<u8>,
    counter: u32,
    count_pat: CountPat,
) -> Result<u32, String> {
    ic_cdk::println!("Chunk Len: {:?}", chunk.len());
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
        let _ = match count_pat {
            CountPat::Pat0 => STATE.with(|s| {
                s.borrow_mut()
                    .tmp_data0
                    .insert(model_name.clone() + "_" + &content_type, tmp_chunks)
            }),
            CountPat::Pat1 => STATE.with(|s| {
                s.borrow_mut()
                    .tmp_data1
                    .insert(model_name.clone() + "_" + &content_type, tmp_chunks)
            }),
            // CountPat::Pat2 => STATE.with(|s| s.borrow_mut().tmp_data2.insert(
            //     model_name.clone() + "_" + &content_type,
            //     tmp_chunks,
            // )),
            // CountPat::Pat3 => STATE.with(|s| s.borrow_mut().tmp_data3.insert(
            //     model_name.clone() + "_" + &content_type,
            //     tmp_chunks,
            // )),
            // CountPat::Pat4 => STATE.with(|s| s.borrow_mut().tmp_data4.insert(
            //     model_name.clone() + "_" + &content_type,
            //     tmp_chunks,
            // )),
            // CountPat::Pat5 => STATE.with(|s| s.borrow_mut().tmp_data5.insert(
            //     model_name.clone() + "_" + &content_type,
            //     tmp_chunks,
            // )),
            // CountPat::Pat6 => STATE.with(|s| s.borrow_mut().tmp_data6.insert(
            //     model_name.clone() + "_" + &content_type,
            //     tmp_chunks,
            // )),
            // CountPat::Pat7 => STATE.with(|s| s.borrow_mut().tmp_data7.insert(
            //     model_name.clone() + "_" + &content_type,
            //     tmp_chunks,
            // )),
            // CountPat::Pat8 => STATE.with(|s| s.borrow_mut().tmp_data8.insert(
            //     model_name.clone() + "_" + &content_type,
            //     tmp_chunks,
            // )),
            // CountPat::Pat9 => STATE.with(|s| s.borrow_mut().tmp_data9.insert(
            //     model_name.clone() + "_" + &content_type,
            //     tmp_chunks,
            // )),
        };
    } else {
        let _ = match count_pat {
            CountPat::Pat0 => STATE.with(|s| {
                s.borrow_mut()
                    .tmp_data0
                    .get_mut(&(model_name.clone() + "_" + &content_type))
                    .unwrap()
                    .chunks
                    .insert(counter, chunk)
            }),
            CountPat::Pat1 => STATE.with(|s| {
                s.borrow_mut()
                    .tmp_data1
                    .get_mut(&(model_name.clone() + "_" + &content_type))
                    .unwrap()
                    .chunks
                    .insert(counter, chunk)
            }),
            // CountPat::Pat2 => STATE.with(|s| s.borrow_mut().tmp_data2.get_mut(
            //     &(model_name.clone() + "_" + &content_type),
            // ).unwrap().chunks.insert(counter, chunk)),
            // CountPat::Pat3 => STATE.with(|s| s.borrow_mut().tmp_data3.get_mut(
            //     &(model_name.clone() + "_" + &content_type),
            // ).unwrap().chunks.insert(counter, chunk)),
            // CountPat::Pat4 => STATE.with(|s| s.borrow_mut().tmp_data4.get_mut(
            //     &(model_name.clone() + "_" + &content_type),
            // ).unwrap().chunks.insert(counter, chunk)),
            // CountPat::Pat5 => STATE.with(|s| s.borrow_mut().tmp_data5.get_mut(
            //     &(model_name.clone() + "_" + &content_type),
            // ).unwrap().chunks.insert(counter, chunk)),
            // CountPat::Pat6 => STATE.with(|s| s.borrow_mut().tmp_data6.get_mut(
            //     &(model_name.clone() + "_" + &content_type),
            // ).unwrap().chunks.insert(counter, chunk)),
            // CountPat::Pat7 => STATE.with(|s| s.borrow_mut().tmp_data7.get_mut(
            //     &(model_name.clone() + "_" + &content_type),
            // ).unwrap().chunks.insert(counter, chunk)),
            // CountPat::Pat8 => STATE.with(|s| s.borrow_mut().tmp_data8.get_mut(
            //     &(model_name.clone() + "_" + &content_type),
            // ).unwrap().chunks.insert(counter, chunk)),
            // CountPat::Pat9 => STATE.with(|s| s.borrow_mut().tmp_data9.get_mut(
            //     &(model_name.clone() + "_" + &content_type),
            // ).unwrap.chunks.insert(counter, chunk)),
        };
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

    chunk_ids.sort();

    let mut chunks = vec![];
    for chunk_id in chunk_ids {
        let count_pat: CountPat = match chunk_id % MAX_COUNT {
            0 => CountPat::Pat0,
            1 => CountPat::Pat1,
            // 2 => CountPat::Pat2,
            // 3 => CountPat::Pat3,
            // 4 => CountPat::Pat4,
            // 5 => CountPat::Pat5,
            // 6 => CountPat::Pat6,
            // 7 => CountPat::Pat7,
            // 8 => CountPat::Pat8,
            // 9 => CountPat::Pat9,
            _ => return Err("Counter is not set".to_string()),
        };

        let tmp_data = match count_pat {
            CountPat::Pat0 => STATE.with(|s| s.borrow().tmp_data0.clone()),
            CountPat::Pat1 => STATE.with(|s| s.borrow().tmp_data1.clone()),
            // CountPat::Pat2 => STATE.with(|s| s.borrow().tmp_data2.clone()),
            // CountPat::Pat3 => STATE.with(|s| s.borrow().tmp_data3.clone()),
            // CountPat::Pat4 => STATE.with(|s| s.borrow().tmp_data4.clone()),
            // CountPat::Pat5 => STATE.with(|s| s.borrow().tmp_data5.clone()),
            // CountPat::Pat6 => STATE.with(|s| s.borrow().tmp_data6.clone()),
            // CountPat::Pat7 => STATE.with(|s| s.borrow().tmp_data7.clone()),
            // CountPat::Pat8 => STATE.with(|s| s.borrow().tmp_data8.clone()),
            // CountPat::Pat9 => STATE.with(|s| s.borrow().tmp_data9.clone()),
        };

        if tmp_data
            .get(&(model_name.clone() + "_" + &content_type))
            .unwrap()
            .chunks
            .get(&chunk_id)
            .is_none()
        {
            return Err("Chunk id does not exist".to_string());
        }
        chunks.extend(
            tmp_data
                .get(&(model_name.clone() + "_" + &content_type))
                .unwrap()
                .chunks
                .get(&chunk_id)
                .unwrap()
                .clone(),
        );
    }

    let result = if exist_llm(model_name.clone()) {
        match content_type.as_str() {
            "weight" => update_llm_weight(model_name.clone(), chunks),
            "tokenizer" => update_llm_tokenizer(model_name.clone(), chunks),
            _ => return Err("Data type is not set".to_string()),
        }
    } else {
        let llm = match content_type.as_str() {
            "weight" => StableBTreeMapLLM(LLM {
                model_name: model_name.clone(),
                weight: Some(chunks.clone()),
                tokenizer: None,
                config: None,
            }),
            "tokenizer" => StableBTreeMapLLM(LLM {
                model_name: model_name.clone(),
                weight: None,
                tokenizer: Some(chunks.clone()),
                config: None,
            }),
            _ => return Err("Data type is not set".to_string()),
        };
        insert_llm(model_name.clone(), llm)
    };

    remove_tmp_data(model_name.clone(), content_type.clone());

    match result {
        Some(model_name) => Ok(model_name),
        None => Err("Failed".to_string()),
    }
}

#[query]
fn exist_llm(key: String) -> bool {
    STATE.with(|s| s.borrow().stable_data.contains_key(&key))
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
fn update_llm_weight(key: String, weight: Vec<u8>) -> Option<String> {
    let llm_result = STATE.with(|s| s.borrow().stable_data.get(&key));
    let mut llm = match llm_result {
        Some(llm) => llm,
        None => return None,
    };
    llm = StableBTreeMapLLM(LLM {
        model_name: llm.0.model_name.clone(),
        weight: Some(weight),
        tokenizer: llm.0.tokenizer.clone(),
        config: llm.0.config.clone(),
    });
    STATE.with(|s| s.borrow_mut().stable_data.insert(key.clone(), llm));
    Some(key)
}

#[update]
fn update_llm_tokenizer(key: String, tokenizer: Vec<u8>) -> Option<String> {
    let llm_result = STATE.with(|s| s.borrow().stable_data.get(&key));
    let mut llm = match llm_result {
        Some(llm) => llm,
        None => return None,
    };
    llm = StableBTreeMapLLM(LLM {
        model_name: llm.0.model_name.clone(),
        weight: llm.0.weight.clone(),
        tokenizer: Some(tokenizer),
        config: llm.0.config.clone(),
    });
    STATE.with(|s| s.borrow_mut().stable_data.insert(key.clone(), llm));
    Some(key)
}

#[update]
pub fn remove_tmp_data(model_name: String, content_type: String) -> Vec<bool> {
    let mut result_vec = vec![];
    for i in 0..MAX_COUNT {
        let count_pat = match i {
            0 => CountPat::Pat0,
            1 => CountPat::Pat1,
            // 2 => CountPat::Pat2,
            // 3 => CountPat::Pat3,
            // 4 => CountPat::Pat4,
            // 5 => CountPat::Pat5,
            // 6 => CountPat::Pat6,
            // 7 => CountPat::Pat7,
            // 8 => CountPat::Pat8,
            // 9 => CountPat::Pat9,
            _ => {
                result_vec.push(false);
                continue;
            }
        };
        match count_pat {
            CountPat::Pat0 => {
                let exist_tmp_chunks = STATE.with(|s| {
                    s.borrow_mut()
                        .tmp_data0
                        .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
                });
                if exist_tmp_chunks {
                    STATE.with(|s| {
                        s.borrow_mut()
                            .tmp_data0
                            .remove(&(model_name.clone() + "_" + &content_type.clone()))
                    });
                    result_vec.push(true);
                } else {
                    result_vec.push(false);
                }
            }
            CountPat::Pat1 => {
                let exist_tmp_chunks = STATE.with(|s| {
                    s.borrow_mut()
                        .tmp_data1
                        .contains_key(&(model_name.clone() + "_" + &content_type.clone()))
                });
                if exist_tmp_chunks {
                    STATE.with(|s| {
                        s.borrow_mut()
                            .tmp_data1
                            .remove(&(model_name.clone() + "_" + &content_type.clone()))
                    });
                    result_vec.push(true);
                } else {
                    result_vec.push(false);
                }
            }
        };
    }
    result_vec
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
