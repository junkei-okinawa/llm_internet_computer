use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use ic_stable_structures::DefaultMemoryImpl;
use std::cell::RefCell;

// A memory for upgrades, where data from the heap can be serialized/deserialized.
const UPGRADES: MemoryId = MemoryId::new(0);

// A memory for the StableBTreeMap we're using. A new memory should be created for
// every additional stable structure.
const STABLE_BTREE: MemoryId = MemoryId::new(1);

// const STALE_TMP0: MemoryId = MemoryId::new(10);
// const STALE_TMP1: MemoryId = MemoryId::new(11);
// pub const STABLE_TMP_COUNT: u32 = 2;

pub type Memory = VirtualMemory<DefaultMemoryImpl>;

// pub enum MemoryType {
//     StaleTmp0,
//     StaleTmp1,
// }

// pub fn get_memory_type(id: u32) -> MemoryType {
//     match id % 10 {
//         0 => MemoryType::StaleTmp0,
//         1 => MemoryType::StaleTmp1,
//         _ => panic!("Invalid memory id"),
//     }
// }

thread_local! {
    // The memory manager is used for simulating multiple memories. Given a `MemoryId` it can
    // return a memory that can be used by stable structures.
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

pub fn get_upgrades_memory() -> Memory {
    MEMORY_MANAGER.with(|m| m.borrow().get(UPGRADES))
}

pub fn get_stable_btree_memory() -> Memory {
    MEMORY_MANAGER.with(|m| m.borrow().get(STABLE_BTREE))
}

// pub fn get_stale_tmp_memory(id: u32) -> Memory {
//     let memory_type = get_memory_type(id);
//     match memory_type {
//         MemoryType::StaleTmp0 => MEMORY_MANAGER.with(|m| m.borrow().get(STALE_TMP0)),
//         MemoryType::StaleTmp1 => MEMORY_MANAGER.with(|m| m.borrow().get(STALE_TMP1)),
//     }
// }
