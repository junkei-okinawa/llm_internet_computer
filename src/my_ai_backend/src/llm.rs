use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use tokenizers::Tokenizer;

use ic_cdk::api::time;
use serde::Deserialize;
use std::io::Read;
use std::path::{Path, PathBuf};

const SEED: u64 = 299792458;

#[derive(Debug)]
pub enum SelectedModel {
    MixFormer(MixFormer),
    Quantized(QMixFormer),
}

#[derive(Debug, Deserialize)]
pub struct ConfigData {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_inner: Option<usize>,
    pub n_head: usize,
    pub rotary_dim: usize,
    pub activation_function: String,
    pub layer_norm_epsilon: f64,
    pub tie_word_embeddings: bool,
    pub pad_vocab_size_multiple: Option<usize>,
}

#[derive(Deserialize)]
struct CustomConfigStruct(Config);

pub trait CustomConfigTrait {
    fn from_file<P: AsRef<Path>>(path: P) -> Self;
    fn from_slice(bytes: &[u8]) -> Self;
}

impl CustomConfigTrait for Config {
    fn from_slice(bytes: &[u8]) -> Self {
        let json_result = serde_json::from_slice(bytes);
        if json_result.is_err() {
            panic!("failed to parse the config file");
        }
        let data: ConfigData = json_result.unwrap();

        let activation_function = match data.activation_function.as_str() {
            "gelu" => Activation::Gelu,
            "new_gelu" => Activation::NewGelu,
            "relu" => Activation::Relu,
            "relu2" => Activation::Relu2,
            "relu6" => Activation::Relu6,
            "silu" => Activation::Silu,
            "sigmoid" => Activation::Sigmoid,
            "hard_sigmoid" => Activation::HardSigmoid,
            "swiglu" => Activation::Swiglu,
            "swish" => Activation::Swish,
            "hard_swish" => Activation::HardSwish,
            "elu" => Activation::Elu(data.layer_norm_epsilon as f64),
            "leaky_relu" => Activation::LeakyRelu(data.layer_norm_epsilon as f64),
            _ => Activation::NewGelu,
        };

        // data key contains "pad_vocab_size_multiple"?
        let pad_vocab_size_multiple = match data.pad_vocab_size_multiple {
            None => 64,
            Some(pad_vocab_size_multiple) => pad_vocab_size_multiple,
        };

        Config::new(
            data.vocab_size,
            data.n_positions,
            data.n_embd,
            data.n_layer,
            data.n_inner,
            data.n_head,
            data.rotary_dim,
            activation_function,
            data.layer_norm_epsilon,
            data.tie_word_embeddings,
            pad_vocab_size_multiple,
        )
    }

    fn from_file<P: AsRef<Path>>(path: P) -> Self {
        // read the file
        let read_result = std::fs::read_to_string(path).unwrap();
        // JSONを中間データ構造にデシリアライズ
        let json_result = serde_json::from_str(&read_result);
        // デシリアライズに失敗したらエラーを返す
        if json_result.is_err() {
            panic!("failed to parse the config file");
        }

        let data: ConfigData = json_result.unwrap();

        let activation_function = match data.activation_function.as_str() {
            "gelu" => Activation::Gelu,
            "new_gelu" => Activation::NewGelu,
            "relu" => Activation::Relu,
            "relu2" => Activation::Relu2,
            "relu6" => Activation::Relu6,
            "silu" => Activation::Silu,
            "sigmoid" => Activation::Sigmoid,
            "hard_sigmoid" => Activation::HardSigmoid,
            "swiglu" => Activation::Swiglu,
            "swish" => Activation::Swish,
            "hard_swish" => Activation::HardSwish,
            "elu" => Activation::Elu(data.layer_norm_epsilon as f64),
            "leaky_relu" => Activation::LeakyRelu(data.layer_norm_epsilon as f64),
            _ => Activation::NewGelu,
        };

        // data key contains "pad_vocab_size_multiple"?
        let pad_vocab_size_multiple = match data.pad_vocab_size_multiple {
            None => 64,
            Some(pad_vocab_size_multiple) => pad_vocab_size_multiple,
        };

        Config::new(
            data.vocab_size,
            data.n_positions,
            data.n_embd,
            data.n_layer,
            data.n_inner,
            data.n_head,
            data.rotary_dim,
            activation_function,
            data.layer_norm_epsilon,
            data.tie_word_embeddings,
            pad_vocab_size_multiple,
        )
    }
}

pub struct Model {
    pub model: SelectedModel,
    pub tokenizer: Tokenizer,
    pub logits_processor: LogitsProcessor,
    pub tokens: Vec<u32>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Model {
    pub fn read_model_file(path: PathBuf) -> Vec<u8> {
        // let path = std::path::Path::new(path);
        if !path.exists() {
            panic!("model file not found. path: {:?}", path);
        }

        let mut file_content = Vec::new();
        let mut file = std::fs::File::open(&path).expect("Unable to open file");
        file.read_to_end(&mut file_content).expect("Unable to read");
        // ic_cdk::api::print(format!("file_content: {:?}", file_content);
        file_content
    }

    pub fn load_from_file(model_dir: PathBuf, quantized: bool) -> Result<Model, E> {
        ic_cdk::api::print(format!("loading model"));
        let start = time();
        let device = &Device::Cpu;
        let config_path = model_dir.join("config.json");
        ic_cdk::api::print(format!("config_path: {:?}", config_path));
        let config = Config::from_file(config_path);
        let tokenizer_path = model_dir.join("tokenizer.json");
        ic_cdk::api::print(format!("tokenizer_path: {:?}", tokenizer_path));
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
        let model = if quantized {
            let weights = model_dir.join("model.gguf");
            ic_cdk::api::print(format!("weights path: {:?}", weights));
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(weights)?;
            let model = QMixFormer::new(&config, vb)?;
            SelectedModel::Quantized(model)
        } else {
            let model_path = model_dir.join("model.safetensors");
            ic_cdk::api::print(format!("weights path: {:?}", model_path));
            let weights = Self::read_model_file(model_path);
            let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
            let model = MixFormer::new(&config, vb)?;
            SelectedModel::MixFormer(model)
        };
        ic_cdk::api::print(format!("loaded the model in {:?}", time() - start));
        let logits_processor = LogitsProcessor::new(SEED, None, None);
        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
        })
    }

    pub fn init_with_prompt(
        &mut self,
        prompt: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
    ) -> Result<String, E> {
        ic_cdk::api::print(format!("initializing the model"));
        ic_cdk::api::print(format!("prompt: {:?}", prompt));
        ic_cdk::api::print(format!("temp: {:?}", temp));
        ic_cdk::api::print(format!("top_p: {:?}", top_p));
        ic_cdk::api::print(format!("repeat_penalty: {:?}", repeat_penalty));
        ic_cdk::api::print(format!("repeat_last_n: {:?}", repeat_last_n));
        ic_cdk::api::print(format!("seed: {:?}", seed));
        match &mut self.model {
            SelectedModel::MixFormer(m) => m.clear_kv_cache(),
            SelectedModel::Quantized(m) => m.clear_kv_cache(),
        };
        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };
        self.logits_processor = LogitsProcessor::new(seed, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.tokens.clear();
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| m.to_string())
            .unwrap()
            .get_ids()
            .to_vec();
        let text = self.process(&tokens).map_err(|m| m.to_string()).unwrap();
        Ok(text)
    }

    pub fn next_token(&mut self) -> Result<String, E> {
        let last_token = *self.tokens.last().unwrap();
        let text = self
            .process(&[last_token])
            .map_err(|m| m.to_string())
            .unwrap();
        Ok(text)
    }

    fn process(&mut self, tokens: &[u32]) -> candle::Result<String> {
        let dev = Device::Cpu;
        let input = Tensor::new(tokens, &dev)?.unsqueeze(0)?;
        let logits = match &mut self.model {
            SelectedModel::MixFormer(m) => m.forward(&input)?,
            SelectedModel::Quantized(m) => m.forward(&input)?,
        };
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);
        let token = match self.tokenizer.decode(&[next_token], false) {
            Ok(token) => token,
            Err(e) => {
                ic_cdk::api::print(format!("error decoding token: {:?}", e));
                "".to_string()
            }
        };
        // ic_cdk::api::print(format!("token: {:?}: {:?}", token, next_token);
        Ok(token)
    }
}
