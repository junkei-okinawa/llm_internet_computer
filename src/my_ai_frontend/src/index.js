import { my_ai_backend } from "../../declarations/my_ai_backend";

document.querySelector("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = e.target.querySelector("button");

  const name = document.getElementById("name").value.toString();

  button.setAttribute("disabled", true);

  // Interact with foo actor, calling the greet method
  const greeting = await my_ai_backend.greet(name);

  button.removeAttribute("disabled");

  document.getElementById("greeting").innerText = greeting;

  return false;
});

let file;
let content_type;
// let fileArrayBuffer;

const input = document.querySelector("section#file>input");
input?.addEventListener("change", (e) => {
  file = e.target.files?.[0];
  if (file.name === "model.safetensors") {
    content_type = "weight";
  } else if (file.name === "tokenizer.json") {
    content_type = "tokenizer";
  } else {
    alert("Please select a valid file");
    file = undefined;
    return;
  }
  console.log("file: ", file);
  console.log("content_type: ", content_type);

});

const uploadChunk = async ({ counter, currentSize, model_name, content_type, chunk }) => {
  console.log("counter: ", counter);
  console.log("currentSize: ", currentSize);

  return my_ai_backend.create_chunk(
    model_name,
    content_type,
    [...new Uint8Array(await chunk.arrayBuffer())],
    counter,
  );
}

const asyncCommitBatch = async (model_name, content_type, ids) => {
  console.log("call commit_batch. ids.length: ", ids.length);
  return my_ai_backend.commit_batch(
    model_name,
    content_type,
    ids,
  );
}

const wait = async (ms) => new Promise(resolve => setTimeout(resolve, ms));

const upload = async () => {
  if (!file) {
    alert("Please select a file");
    return;
  }

  if (!content_type) {
    alert("Please select a file");
    return;
  }

  const model_name = document.querySelector("input#model").value.toString();
  if (!model_name) {
    alert("Please input a model name");
    return;
  }

  console.log("start upload");

  const chunkSize = 1024 * 1024 * 1.9; // 1.9MB
  const splitCount = 100;
  const ids = [];
  const collectedIds = [];


  let chunks;
  let chunkIds;
  let counter = 0;
  let currentSize = 0;
  let i = 0
  while (i < file.size) {
    chunks = [];
    chunkIds = [];
    for (i; i < file.size; i += chunkSize) {
      counter++;
      currentSize = i + chunkSize > file.size ? file.size : i + chunkSize;
      const chunk = file.slice(i, i + chunkSize);
      chunks.push(uploadChunk({ counter, currentSize, model_name, content_type, chunk }));
      if (i !== 0 && counter % splitCount === 0) {
        break;
      }
    }

    console.log("chunks.length: ", chunks.length);
    console.log("chunks: ", chunks);

    if (chunks.length === 0) {
      break;
    }

    chunkIds = await Promise.all(chunks);
    collectedIds.push(chunkIds.length);
    console.log("!!!!!!! Collect IDs !!!!!!! chunkIds: ", chunkIds);
    console.log("!!!! collectedIds.length: ", collectedIds.length);

    for (let k = 0; k < chunkIds.length; k++) {
      console.log("push ids")
      ids.push(chunkIds[k].Ok);
    }
  }
  file = null;
  chunks = null;

  console.log("call commit_batch. ids.length: ", ids.length);

  if (ids.length === 0) {
    alert("Please select a file");
    return;
  }

  console.log("ids: ", ids);

  const result = await my_ai_backend.commit_batch(
    model_name,
    content_type,
    ids,
  );

  console.log("file uploaded");
  console.log("result: ", result);
  content_type = null;
}

const buttonUpload = document.querySelector("button.upload");
buttonUpload?.addEventListener("click", upload);