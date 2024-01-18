import { JSOX } from "jsox";
import { Principal } from '@dfinity/principal';
import { AuthClient } from '@dfinity/auth-client';
import { AssetManager } from '@dfinity/assets'
import { HttpAgent } from '@dfinity/agent';
import { createActor } from "../../declarations/my_ai_backend";
import { canisterId } from "../../declarations/my_ai_backend";
console.log("canisterId: ", canisterId);

// ===================
const signInBtn = document.getElementById('signinBtn');
const signOutBtn = document.getElementById('signoutBtn');
const principalEl = document.getElementById('principal');
const idpUrlEl = document.getElementById('idpUrl');
const hostUrlEl = document.getElementById('hostUrl');
const canisterIdEl = document.getElementById('canisterId');
canisterIdEl.value = canisterId;

const permissionEl = document.getElementById('permission');
console.log("permissionEl.value: ", permissionEl.value);

const uploadResultEl = document.getElementById('uploadResult');
const progressEl = document.getElementById('fileUploadProgress');

if (process.env.DFX_NETWORK == "local") {
  const providerUrl = `http://${process.env.CANISTER_ID_INTERNET_IDENTITY}.localhost:8000/`;
  idpUrlEl.value = providerUrl;
}
let authClient;

const init = async () => {
  authClient = await AuthClient.create();
  principalEl.innerText = await authClient.getIdentity().getPrincipal();

  // Redirect to the identity provider
  signInBtn.onclick = async () => {
    authClient.login({
      identityProvider: idpUrlEl.value,
      maxTimeToLive: 8.64 * 10 ** 13 * 7, // one week
      onSuccess: async () => {
        principalEl.innerText = await authClient.getIdentity().getPrincipal();
      },
    });
  };

  signOutBtn.onclick = async () => {
    await authClient.logout();
    principalEl.innerText = await authClient.getIdentity().getPrincipal();
  };
};

init();

const backendAssetManager = async (arg) => {
  if (!arg) {
    arg = {};
  }
  const identity = await authClient.getIdentity();
  console.log("identity: ", identity);

  const agentOptions = {
    host: hostUrlEl.value,
    identity
  }
  const agent = new HttpAgent(agentOptions);
  if (process.env.DFX_NETWORK === "local") {
    agent.fetchRootKey();
  }
  return new AssetManager({ canisterId, agent, ...arg });
}

const grantPermission = async () => {
  if (permissionEl.value === "") {
    alert("Please select permission");
    return;
  }
  const permission = {};
  permission[permissionEl.value] = null;
  const to_principal = Principal.fromText(principalEl.innerText);
  const identity = await authClient.getIdentity();
  console.log("identity: ", identity);
  const agentOptions = {
    host: hostUrlEl.value,
    identity
  }
  const my_ai_backend = createActor(canisterId, { agentOptions });
  const result = await my_ai_backend.grant_permission({ permission, to_principal });
  console.log("result: ", result);
}

const buttonGrantPermission = document.querySelector("button#grantPermission");
buttonGrantPermission?.addEventListener("click", grantPermission);

let file;
let content_type;
const input = document.querySelector("section#file>input");
input?.addEventListener("change", (e) => {
  file = e.target.files?.[0];
  if (file.name === "model.safetensors") {
    content_type = "weight";
  } else if (file.name === "tokenizer.json") {
    content_type = "tokenizer";
  } else if (file.name.indexOf("config.json") !== -1) {
    content_type = "config";
  } else if (file.name.indexOf(".gguf") !== -1) {
    content_type = "gguf";
  } else {
    alert("Please select a valid file");
    file = undefined;
    return;
  }
  console.log("file: ", file);
  console.log("content_type: ", content_type);

});

const upload = async () => {
  console.log("start upload");
  uploadResultEl.innerText = "";
  if (file.size > 1024 * 1024 * 1.9) {
    await chunkUploader();
    return;
  }

  const asset_buffer = await file.arrayBuffer();
  console.log("asset_buffer: ", asset_buffer);
  const content = new Uint8Array(asset_buffer);
  console.log("content: ", content);
  console.log("file.name: ", file.name);
  console.log("file.type: ", file.type);
  const content_encoding = "identity";
  console.log("content_encoding: ", content_encoding);

  const sha256 = [];
  const aliased = [];
  // ================================

  const my_ai_backend = await backendAssetManager();
  uploadResultEl.innerText = "Uploading...";
  progressEl.removeAttribute("hidden");
  progressEl.max = file.size;
  progressEl.value = 0;

  const config = {
    fileName: file.name,
    contentType: file.type,
    onProgress: (progress) => {
      console.log("progress: ", progress);
      uploadResultEl.innerText = "Uploading... progress: " + JSON.stringify(progress);
      progressEl.max = progress.total;
      progressEl.value = progress.current;
    },
  };
  const result = await my_ai_backend.store(file, config);
  console.log("result: ", result);

  uploadResultEl.innerText = "Uploaded fileName: " + file.name;
  progressEl.setAttribute("hidden", "hidden");
  file = null;
  await getAssetsList();
}

const chunkUploader = async () => {
  console.log("start chunkUploader");
  // ================================
  uploadResultEl.innerText = "";
  // console.log("progressEl: ", progressEl);
  progressEl.removeAttribute("hidden");
  progressEl.max = file.size;
  progressEl.value = 0;

  const my_ai_backend = await backendAssetManager();
  const batchManager = my_ai_backend.batch();
  console.log("batchManager: ", batchManager);

  const config = {
    fileName: file.name,
    contentType: file.type,
    onProgress: (progress) => {
      console.log("progress: ", progress);
      uploadResultEl.innerText = "Uploading... progress: " + JSON.stringify(progress);
      progressEl.max = progress.total;
      progressEl.value = progress.current;
    },
  };
  const key = await batchManager.store(file, config);
  console.log("key: ", key);

  const commitBatchArgs = {
    fileName: key,
    onProgress: (progress) => {
      console.log("progress: ", progress);
      uploadResultEl.innerText = "Commiting... progress: " + JSON.stringify(progress);
      progressEl.max = progress.total;
      progressEl.value = progress.current;
    },
  }
  const result = await batchManager.commit({ ...commitBatchArgs });
  console.log("result: ", result);

  uploadResultEl.innerText = "Uploaded fileName: " + file.name;
  progressEl.setAttribute("hidden", "hidden");
  // ================================

  console.log("file uploaded");
  file = null;
  await getAssetsList();
}

const buttonUpload = document.querySelector("button.upload");
buttonUpload?.addEventListener("click", upload);

const getAssetsList = async () => {
  console.log("start getAssetsList");
  const my_ai_backend = await backendAssetManager();

  const files = await my_ai_backend.list();
  console.log("files: ", files);
  const assetsListElement = document.querySelector("section#assetsList>pre");
  assetsListElement.innerText = JSOX.stringify(files, null, 2);
}

const buttonOutputJson = document.querySelector("button#getAssetsList");
buttonOutputJson?.addEventListener("click", getAssetsList);

const clear = async () => {
  const my_ai_backend = await backendAssetManager();

  const result = await my_ai_backend.clear();
  console.log("result: ", result);
  await getAssetsList();
}

const buttonClear = document.querySelector("button#clear");
buttonClear?.addEventListener("click", clear);

const loadConfig = async () => {
  const inputLoadConfigEl = document.querySelector("input#loadConfig");
  console.log("inputLoadConfigEl.value: ", inputLoadConfigEl.value);
  if (inputLoadConfigEl.value === "") {
    alert("Please input Load Config Path.");
    return;
  }
  const identity = await authClient.getIdentity();
  console.log("identity: ", identity);

  const agentOptions = {
    host: hostUrlEl.value,
    identity
  }
  const my_ai_backend = createActor(canisterId, { agentOptions });
  const argLoadConfig = { 'key': inputLoadConfigEl.value, 'accept_encodings': ["identity"] };
  const result = await my_ai_backend.load_config(argLoadConfig);
  console.log("result: ", result);
}

const buttonLoadConfig = document.querySelector("button#loadConfig");
buttonLoadConfig?.addEventListener("click", loadConfig);

const loadTokenizer = async () => {
  const inputLoadTokenizerEl = document.querySelector("input#loadTokenizer");
  console.log("inputLoadTokenizerEl.value: ", inputLoadTokenizerEl.value);
  if (inputLoadTokenizerEl.value === "") {
    alert("Please input Load Tokenizer Path.");
    return;
  }
  const identity = await authClient.getIdentity();
  console.log("identity: ", identity);

  const agentOptions = {
    host: hostUrlEl.value,
    identity
  }
  const my_ai_backend = createActor(canisterId, { agentOptions });
  const argLoadTokenizer = { 'key': inputLoadTokenizerEl.value, 'accept_encodings': ["identity"] };
  const result = await my_ai_backend.load_tokenizer(argLoadTokenizer);
  console.log("result: ", result);
}

const buttonLoadTokenizer = document.querySelector("button#loadTokenizer");
buttonLoadTokenizer?.addEventListener("click", loadTokenizer);

const loadModel = async () => {
  const inputLoadModelEl = document.querySelector("input#loadModel");
  console.log("inputLoadModelEl.value: ", inputLoadModelEl.value);
  if (inputLoadModelEl.value === "") {
    alert("Please input Load Model Path.");
    return;
  }
  const identity = await authClient.getIdentity();
  console.log("identity: ", identity);

  const agentOptions = {
    host: hostUrlEl.value,
    identity
  }
  const my_ai_backend = createActor(canisterId, { agentOptions });
  const argLoadModel = { 'key': inputLoadModelEl.value, 'accept_encodings': ["identity"] };
  const result = await my_ai_backend.load_model(argLoadModel);
  console.log("result: ", result);
}

const buttonLoadModel = document.querySelector("button#loadModel");
buttonLoadModel?.addEventListener("click", loadModel);

const loadGGUF = async () => {
  const inputLoadGGUFEl = document.querySelector("input#loadGGUF");
  console.log("inputLoadGGUFEl.value: ", inputLoadGGUFEl.value);
  if (inputLoadGGUFEl.value === "") {
    alert("Please input Load GGUF Path.");
    return;
  }
  const identity = await authClient.getIdentity();
  console.log("identity: ", identity);

  const agentOptions = {
    host: hostUrlEl.value,
    identity
  }
  const my_ai_backend = createActor(canisterId, { agentOptions });
  const argLoadGGUF = { 'key': inputLoadGGUFEl.value, 'accept_encodings': ["identity"] };
  const result = await my_ai_backend.load_gguf(argLoadGGUF);
  console.log("result: ", result);
}

const buttonLoadGGUF = document.querySelector("button#loadGGUF");
buttonLoadGGUF?.addEventListener("click", loadGGUF);

const setModelContent = async () => {
  const inputSetModelContentEl = document.querySelector("input#setModelContent");
  console.log("inputSetModelContentEl.value: ", inputSetModelContentEl.value);
  if (inputSetModelContentEl.value === "") {
    alert("Please input Set Model Content Path.");
    return;
  }
  const identity = await authClient.getIdentity();
  console.log("identity: ", identity);

  const agentOptions = {
    host: hostUrlEl.value,
    identity
  }
  const my_ai_backend = createActor(canisterId, { agentOptions });
  const argSetModelContent = {
    'key': inputSetModelContentEl.value,
    'content_encoding': "identity",
    'set_content_count': 90,
    'now': Date.now()
  };
  progressEl.max = 0;
  progressEl.removeAttribute("hidden");
  while (true) {
    const result = await my_ai_backend.set_model_content(argSetModelContent);
    console.log("result: ", result);
    uploadResultEl.innerText = "Model Setting... progress: {current_chunks_length: " + result.Ok.current_chunks_length + ", total_chunks_length: " + result.Ok.total_chunks_length + "}";
    progressEl.max = result.Ok.total_chunks_length;
    progressEl.value = result.Ok.current_chunks_length;
    if (result.Ok.total_chunks_length === result.Ok.current_chunks_length) {
      console.log("break");
      break;
    }
  }
  uploadResultEl.innerText = "Model Setted!";
  progressEl.setAttribute("hidden", "hidden");
}

const buttonSetModelContent = document.querySelector("button#setModelContent");
buttonSetModelContent?.addEventListener("click", setModelContent);

const getAsset = async () => {
  const inputGetAssetEl = document.querySelector("input#getAsset");
  console.log("inputGetAssetEl.value: ", inputGetAssetEl.value);
  if (inputGetAssetEl.value === "") {
    alert("Please input Get Asset Path.");
    return;
  }
  const assetManagerArgs = { maxSingleFileSize: 1024 * 1024 * 1024 * 10 } // 10GB
  const my_ai_backend = await backendAssetManager(assetManagerArgs);
  const result_asset = await my_ai_backend.get(inputGetAssetEl.value);
  console.log("result_asset: ", result_asset);
  const blob = await result_asset.toBlob();
  const link = document.createElement('a');
  const fileName = inputGetAssetEl.value.split("/").pop();
  link.download = fileName; // ダウンロードファイル名称
  link.href = URL.createObjectURL(blob); // オブジェクト URL を生成
  link.click(); // クリックイベントを発生させる
  URL.revokeObjectURL(link.href); // オブジェクト URL を解放」
}

const buttonGetAsset = document.querySelector("button#getAsset");
buttonGetAsset?.addEventListener("click", getAsset);


// dfx ledger fabricate-cycles --canister my_ai_backend --cycles 500000000000000