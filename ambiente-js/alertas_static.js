/**
 * alertas_static.js (versão sem fallback automático)
 *
 * - Roda ONNX pose_alert_box.onnx
 * - Filtra por modo APÓS a inferência
 * - Encerra se o modo informado for inválido
 */

const fs = require("fs");
const ort = require("onnxruntime-node");
const { createCanvas, loadImage } = require("canvas");
const sharp = require("sharp");
const readline = require("readline");

// ====== Config de caminhos ======
const imagePath = "./test-front-640.png";
const jsonPath = "./yolo_filtered.json";
const modelPath = "/home/inovia/Documentos/rep-app-pose/pose_alert_box.onnx";
const outputPath = "./imagem_com_alerta.png";

// ====== Configs ======
const largura = 640;
const altura = 640;
const confThreshold = 0.1;
const PROB_THRESHOLD = 0.5;

// ====== Nomes dos 14 alertas ======
const ALERT_NAMES = [
  "Afaste as pernas",                      // 0
  "Mais de uma pessoa detectada",          // 1
  "Abaixe os braços",                      // 2
  "Abra os braços",                        // 3
  "Afaste-se da câmera (margem YOLO bbox)",// 4
  "Aproxime-se da câmera",                 // 5
  "Vire de costas",                        // 6
  "Estique os braços na altura do ombro",  // 7
  "Vire para a direita",                   // 8
  "Estique os braços na altura do ombro",  // 9
  "Vire para a esquerda",                  // 10
  "Afaste-se (pontos virtuais)",           // 11
  "Fique de frente com a câmera",          // 12
  "Aproxime-se (bbox pequena YOLO)",       // 13
];

// ====== Filtros por modo ======
const MODO_ALERTAS = {
  front: [0, 1, 2, 3, 4, 12, 13],
  back:  [0, 1, 2, 3, 4, 6],
  right: [1, 7, 8, 4, 13],
  left:  [1, 9, 10, 4],
};

// ====== Utils ======
function scalar32(x) {
  return new ort.Tensor("float32", new Float32Array([x]), []);
}

function modoOneHotArray(modo) {
  switch (modo) {
    case "front": return [1, 0, 0, 0];
    case "back": return [0, 1, 0, 0];
    case "right": return [0, 0, 1, 0];
    case "left": return [0, 0, 0, 1];
    default: return [1, 0, 0, 0];
  }
}

function askPoseMode() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question("Selecione o modo de pose ('front','back','right','left'): ", (answer) => {
      rl.close();
      const modo = (answer || "").trim().toLowerCase();
      if (!MODO_ALERTAS[modo]) {
        console.error(`Modo inválido: "${modo}". Use um dos seguintes: front, back, right, left.`);
        process.exit(1); // encerra imediatamente
      } else resolve(modo);
    });
  });
}

// ====== Main ======
async function main() {
  const modo = await askPoseMode();

  // 1) Imagem base
  const imgBuffer = await sharp(imagePath).resize(largura, altura).toBuffer();
  const baseImage = await loadImage(imgBuffer);
  const canvas = createCanvas(largura, altura);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(baseImage, 0, 0, largura, altura);

  // 2) JSON detecções
  const json = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  const detections = json.detections;
  if (!detections || detections.length === 0) throw new Error("Nenhuma detecção encontrada no JSON");

  // 3) Melhor detecção
  const best = detections.reduce((a, b) => (a.confidence > b.confidence ? a : b));
  console.log(`✔️ Usando detecção com confiança: ${best.confidence.toFixed(3)}`);

  const keypoints = best.keypoints;
  const bbox = best.bbox || (best.x1 !== undefined ? [best.x1, best.y1, best.x2, best.y2] : null);
  if (!bbox) console.warn(" BBox não encontrada — desenho da bbox será ignorado.");

  // 4) Normaliza keypoints para [0,1]
  const kpsNormFlat = Float32Array.from(keypoints.flatMap(([x, y, c]) => [x / largura, y / altura, c]));

  // 5) Sessão ONNX
  const session = await ort.InferenceSession.create(modelPath);
  console.log("Model inputs:", session.inputNames);

  const inputNamesLower = session.inputNames.map(n => n.toLowerCase());
  const modoInputName = session.inputNames.find(n => /modo|mode|pose_mode/i.test(n));
  const acceptsModo = !!modoInputName;

  // 6) Feeds
  const feeds = {};
  feeds[session.inputNames[0]] = new ort.Tensor("float32", kpsNormFlat, [1, 17, 3]);
  if (inputNamesLower.includes("image_width"))
    feeds[session.inputNames[inputNamesLower.indexOf("image_width")]] = scalar32(largura);
  if (inputNamesLower.includes("image_height"))
    feeds[session.inputNames[inputNamesLower.indexOf("image_height")]] = scalar32(altura);
  if (inputNamesLower.includes("num_pessoas_tensor"))
    feeds[session.inputNames[inputNamesLower.indexOf("num_pessoas_tensor")]] = scalar32(1.0);
  if (inputNamesLower.includes("yolo_bboxes")) {
    const val = bbox ? Float32Array.from(bbox) : new Float32Array([0, 0, 0, 0]);
    feeds[session.inputNames[inputNamesLower.indexOf("yolo_bboxes")]] = new ort.Tensor("float32", val, [1, 4]);
  }

  if (acceptsModo) {
    console.log("Modelo aceita input de modo:", modoInputName);
    feeds[modoInputName] = new ort.Tensor("float32", Float32Array.from(modoOneHotArray(modo)), [1, 4]);
  } else {
    console.log("Modelo NÃO aceita input de modo. Aplicando lógica local.");
  }

  // 7) Inferência
  let results;
  try {
    results = await session.run(feeds);
  } catch (e) {
    console.error(" Erro na inferência:", e.message);
    throw e;
  }

  // 8) Saída (14 valores)
  const outputTensor = Object.values(results)[0];
  const alertVals = Array.from(outputTensor.data);
  console.log("🔍 Saída do modelo (raw):", alertVals.map(v => v.toFixed(3)));

  // 9) Lista completa
  const alertasCompletos = ALERT_NAMES.map((nome, i) => ({
    nome,
    ativo: alertVals[i] > PROB_THRESHOLD
  }));

  // 10) Filtra alertas do modo
  const indicesModo = MODO_ALERTAS[modo];
  const alertasModo = indicesModo.map(i => alertasCompletos[i]);
  const ativos = alertasModo.filter(a => a.ativo).map(a => a.nome);

  // 11) Se não houver alerta ativo → Postura OK
  const texto = ativos.length ? ativos.join(" | ") : "Postura OK";
  const cor = ativos.length ? "red" : "green";

  // 12) Desenha keypoints
  for (const [x, y, conf] of keypoints) {
    if (conf >= confThreshold) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "blue";
      ctx.fill();
    }
  }

  // 13) Desenha bbox
  if (bbox) {
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    const h = bbox[3] - bbox[1];
    const m = Math.max(8, Math.round(0.1 * h));
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(m, 0); ctx.lineTo(m, altura);
    ctx.moveTo(largura - m, 0); ctx.lineTo(largura - m, altura);
    ctx.stroke();
  }

  // 14) Texto final
  ctx.font = "20px sans-serif";
  ctx.fillStyle = cor;
  const linhas = texto.match(/.{1,60}(?:\s|$)/g) || ["Postura OK"];
  linhas.forEach((linha, i) => ctx.fillText(linha.trim(), 20, 40 + i * 28));

  // 15) Salva imagem
  const outBuffer = canvas.toBuffer("image/png");
  fs.writeFileSync(outputPath, outBuffer);
  console.log(`Imagem final salva em: ${outputPath}`);
  console.log(" Mensagem:", texto);
}

main().catch(err => console.error("Erro:", err?.message || err));
