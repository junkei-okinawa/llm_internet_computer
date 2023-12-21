use candid::{CandidType, Decode, Encode};
use ic_cdk_macros::{export_candid, post_upgrade, pre_upgrade, query, update};
use ic_stable_structures::storable::Bound;
use ic_stable_structures::{writer::Writer, Memory as _, StableBTreeMap, Storable};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
// use std::collections::HashMap;
use std::{
    borrow::Cow,
    cell::RefCell,
    cmp::{Eq, Ord, PartialEq, PartialOrd},
};
mod memory;
use memory::Memory;
use memory::STABLE_TMP_COUNT;

const MAX_SIZE: u32 = u32::MAX; // 4GB
const COUNT_SIZE: u32 = STABLE_TMP_COUNT;

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
        max_size: MAX_SIZE,
        is_fixed_size: false,
    };
}

// #[derive(Eq, Ord, PartialEq, PartialOrd, CandidType, Serialize, Deserialize, Debug, Clone)]
// struct TmpData {
//     data: BTreeMap<String, TmpChunks>, // key: model_name & "_" & content_type
// }

#[derive(Eq, Ord, PartialEq, PartialOrd, CandidType, Serialize, Deserialize, Debug, Clone)]
struct TmpChunks {
    chunks: BTreeMap<u32, Vec<u8>>, // key: chunk_id & value: chunk
    data_name: String,
    content_type: ContentType,
}

#[derive(Eq, Ord, PartialEq, PartialOrd, CandidType, Serialize, Deserialize, Debug, Clone)]
enum ContentType {
    Weight,
    Tokenizer,
}

#[derive(CandidType, Ord, PartialOrd, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct StableBTreeMapTmpData(TmpChunks);

impl Storable for StableBTreeMapTmpData {
    fn to_bytes(&self) -> std::borrow::Cow<[u8]> {
        Cow::Owned(Encode!(self).unwrap())
    }
    fn from_bytes(bytes: Cow<'_, [u8]>) -> Self {
        Decode!(&bytes, Self).unwrap()
    }
    const BOUND: Bound = Bound::Bounded {
        max_size: MAX_SIZE,
        is_fixed_size: false,
    };
}

#[derive(Serialize, Deserialize, Clone, CandidType)]
pub enum CountPat {
    Pat0 = 0,
    Pat1 = 1,
    // Pat2 = 2,
    // Pat3 = 3,
    // Pat4 = 4,
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
    #[serde(skip, default = "init_stable_tmp_data0")]
    tmp_data0: StableBTreeMap<String, StableBTreeMapTmpData, Memory>, // key: model_name & "_" & content_type
    #[serde(skip, default = "init_stable_tmp_data1")]
    tmp_data1: StableBTreeMap<String, StableBTreeMapTmpData, Memory>,
    // tmp_data1: TmpData,
    // tmp_data2: TmpData,
    // tmp_data3: TmpData,
    // tmp_data4: TmpData,
}

impl Default for State {
    fn default() -> Self {
        Self {
            data_on_the_heap: vec![],
            stable_data: init_stable_data(),
            tmp_data0: init_stable_tmp_data0(),
            tmp_data1: init_stable_tmp_data1(),
            // tmp_data1: TmpData {
            //     data: HashMap::new(),
            // },
            // tmp_data2: TmpData {
            //     data: HashMap::new(),
            // },
            // tmp_data3: TmpData {
            //     data: HashMap::new(),
            // },
            // tmp_data4: TmpData {
            //     data: HashMap::new(),
            // },
        }
    }
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn get_count_pat(chank_id: u32) -> CountPat {
    match chank_id % COUNT_SIZE {
        0 => CountPat::Pat0,
        1 => CountPat::Pat1,
        // 2 => CountPat::Pat2,
        // 3 => CountPat::Pat3,
        // 4 => CountPat::Pat4,
        _ => panic!("Chunk ID is not set"),
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
        // CountPat::Pat2 => STATE.with(|s| s.borrow().tmp_data2.contains_key(&(model_name.clone() + "_" + &content_type))),
        // CountPat::Pat3 => STATE.with(|s| s.borrow().tmp_data3.contains_key(&(model_name.clone() + "_" + &content_type))),
        // CountPat::Pat4 => STATE.with(|s| s.borrow().tmp_data4.contains_key(&(model_name.clone() + "_" + &content_type))),
    }
}

fn remove_tmp_data(count_pat: CountPat, model_name: String, content_type: String) -> bool {
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data0
                .remove(&(model_name.clone() + "_" + &content_type))
                .is_some()
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data1
                .remove(&(model_name.clone() + "_" + &content_type))
                .is_some()
        }), // CountPat::Pat2 => STATE.with(|s| s.borrow_mut().tmp_data2.remove(&(model_name.clone() + "_" + &content_type)).is_some()),
            // CountPat::Pat3 => STATE.with(|s| s.borrow_mut().tmp_data3.remove(&(model_name.clone() + "_" + &content_type)).is_some()),
            // CountPat::Pat4 => STATE.with(|s| s.borrow_mut().tmp_data4.remove(&(model_name.clone() + "_" + &content_type)).is_some()),
    }
}

fn get_tmp_chunks(count_pat: CountPat, model_name: String, content_type: String) -> TmpChunks {
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow()
                .tmp_data0
                .get(&(model_name.clone() + "_" + &content_type))
                .unwrap()
                .0
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow()
                .tmp_data1
                .get(&(model_name.clone() + "_" + &content_type))
                .unwrap()
                .0
        }),
        // CountPat::Pat2 => STATE.with(|s| s.borrow().tmp_data2.get(&(model_name.clone() + "_" + &content_type)).unwrap().0),
        // CountPat::Pat3 => STATE.with(|s| s.borrow().tmp_data3.get(&(model_name.clone() + "_" + &content_type)).unwrap().0),
        // CountPat::Pat4 => STATE.with(|s| s.borrow().tmp_data4.get(&(model_name.clone() + "_" + &content_type)).unwrap().0),
    }
}

fn insert_tmp_chunks(
    count_pat: CountPat,
    model_name: String,
    content_type: String,
    tmp_chunks: TmpChunks,
) -> Option<StableBTreeMapTmpData> {
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow_mut().tmp_data0.insert(
                model_name.clone() + "_" + &content_type,
                StableBTreeMapTmpData(tmp_chunks),
            )
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow_mut().tmp_data1.insert(
                model_name.clone() + "_" + &content_type,
                StableBTreeMapTmpData(tmp_chunks),
            )
        }),
        // CountPat::Pat2 => STATE.with(|s| {
        //     s.borrow_mut()
        //         .tmp_data2
        //         .insert(model_name.clone() + "_" + &content_type, StableBTreeMapTmpData(tmp_chunks))
        // }),
        // CountPat::Pat3 => STATE.with(|s| {
        //     s.borrow_mut()
        //         .tmp_data3
        //         .insert(model_name.clone() + "_" + &content_type, StableBTreeMapTmpData(tmp_chunks))
        // }),
        // CountPat::Pat4 => STATE.with(|s| {
        //     s.borrow_mut()
        //         .tmp_data4
        //         .insert(model_name.clone() + "_" + &content_type, StableBTreeMapTmpData(tmp_chunks))
        // }),
    }
}

fn update_tmp_chunks(
    count_pat: CountPat,
    model_name: String,
    content_type: String,
    tmp_chunks: TmpChunks,
) {
    match count_pat {
        CountPat::Pat0 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data0
                .get(&(model_name.clone() + "_" + &content_type))
                .unwrap()
                .0 = tmp_chunks
        }),
        CountPat::Pat1 => STATE.with(|s| {
            s.borrow_mut()
                .tmp_data1
                .get(&(model_name.clone() + "_" + &content_type))
                .unwrap()
                .0 = tmp_chunks
        }),
        // CountPat::Pat2 => {
        //     STATE.with(|s| s.borrow_mut().tmp_data2.get(&(model_name.clone() + "_" + &content_type)).unwrap().0 = tmp_chunks)
        // }
        // CountPat::Pat3 => {
        //     STATE.with(|s| s.borrow_mut().tmp_data3.get(&(model_name.clone() + "_" + &content_type)).unwrap().0 = tmp_chunks)
        // }
        // CountPat::Pat4 => {
        //     STATE.with(|s| s.borrow_mut().tmp_data4.get(&(model_name.clone() + "_" + &content_type)).unwrap().0 = tmp_chunks)
        // }
    };
}

#[update]
pub async fn create_chunk(
    model_name: String,
    content_type: String,
    chunk: Vec<u8>,
    counter: u32,
    count_pat: CountPat,
) -> Result<u32, String> {
    ic_cdk::println!("Chunk Len: {:?}, Counter: {:?}", chunk.len(), counter);
    if exist_tmp_data(count_pat.clone(), model_name.clone(), content_type.clone()) {
        let mut tmp_chunks =
            get_tmp_chunks(count_pat.clone(), model_name.clone(), content_type.clone());
        if tmp_chunks.chunks.contains_key(&counter) {
            return Err(format!(
                "Chunk ID is already set. Target Chunk ID {}",
                counter
            ));
        } else {
            tmp_chunks.chunks.insert(counter, chunk);
            update_tmp_chunks(count_pat, model_name, content_type, tmp_chunks);
        }
    } else {
        let mut tmp_chunks = TmpChunks {
            chunks: BTreeMap::new(),
            data_name: model_name.clone(),
            content_type: ContentType::Weight,
        };
        tmp_chunks.chunks.insert(counter, chunk);
        insert_tmp_chunks(count_pat, model_name, content_type, tmp_chunks);
    };

    ic_cdk::println!("Chunk ID: {:?}", counter);
    Ok(counter)
}

#[update]
pub async fn commit_batch(
    model_name: String,
    content_type: String,
    mut chunk_ids: Vec<u32>,
) -> Result<String, String> {
    ic_cdk::println!(
        "start commit_batch. model_name: {}, content_type{}, Chunk IDs: {:?}",
        model_name,
        content_type,
        chunk_ids
    );
    chunk_ids.sort();

    let mut chunks = vec![];
    for chunk_id in chunk_ids {
        let count_pat = get_count_pat(chunk_id);
        if exist_tmp_data(count_pat.clone(), model_name.clone(), content_type.clone()) {
            let tmp_chunks =
                get_tmp_chunks(count_pat.clone(), model_name.clone(), content_type.clone());
            match tmp_chunks.chunks.get(&chunk_id) {
                Some(chunk) => chunks.extend(chunk),
                None => return Err(format!("Chunk ID is not set. Target Chunk ID {}", chunk_id)),
            };
        } else {
            return Err(format!("Chunk ID is not set. Target Chunk ID {}", chunk_id));
        };
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

    remove_target_tmp_data(model_name.clone(), content_type.clone());

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
pub fn remove_target_tmp_data(model_name: String, content_type: String) -> Vec<bool> {
    let mut result = vec![];
    for i in 0..COUNT_SIZE {
        let count_pat = get_count_pat(i);
        let result_exist =
            exist_tmp_data(count_pat.clone(), model_name.clone(), content_type.clone());
        if !result_exist {
            result.push(false);
        } else {
            let result_remove =
                remove_tmp_data(count_pat.clone(), model_name.clone(), content_type.clone());
            result.push(result_remove);
        }
    }
    result
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

fn init_stable_tmp_data0() -> StableBTreeMap<String, StableBTreeMapTmpData, Memory> {
    StableBTreeMap::init(crate::memory::get_stale_tmp_memory(0))
}
fn init_stable_tmp_data1() -> StableBTreeMap<String, StableBTreeMapTmpData, Memory> {
    StableBTreeMap::init(crate::memory::get_stale_tmp_memory(1))
}

#[query]
fn greet(name: String) -> String {
    format!("Hello, {}!", name)
}

export_candid!();
