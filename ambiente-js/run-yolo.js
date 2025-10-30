const fs = require('fs');
const { PNG } = require('pngjs');
const ort = require('onnxruntime-node');

const imagePath = './test-front-640.png';
const modelPath = '/home/inovia/Documentos/rep-app-pose/yolo11n-pose.onnx';
const outputRawJson = 'yolo_raw.json';
const outputFilteredJson = 'yolo_filtered.json';
const confThreshold = 0.25;

// normalizar PNG
function loadPngAsTensor(path) {
  return new Promise((resolve, reject) => {
    fs.createReadStream(path)
      .pipe(new PNG())
      .on('parsed', function () {
        const { width, height, data } = this;
        const tensor = new Float32Array(3 * height * width);
        for (let i = 0; i < height; i++) {
          for (let j = 0; j < width; j++) {
            const idx = (i * width + j) << 2;
            const r = data[idx] / 255;
            const g = data[idx + 1] / 255;
            const b = data[idx + 2] / 255;
            const pixel = i * width + j;
            tensor[0 * height * width + pixel] = r;
            tensor[1 * height * width + pixel] = g;
            tensor[2 * height * width + pixel] = b;
          }
        }
        resolve({ tensor, shape: [1, 3, height, width] });
      })
      .on('error', reject);
  });
}

async function main() {
  const { tensor, shape } = await loadPngAsTensor(imagePath);
  const session = await ort.InferenceSession.create(modelPath);
  const inputName = session.inputNames[0];
  const feeds = {};
  feeds[inputName] = new ort.Tensor('float32', tensor, shape);

  const output = await session.run(feeds);
  const outputTensor = output[session.outputNames[0]];

  // Salva JSON bruto
  const flat = Array.from(outputTensor.data);
  const jsonRaw = {
    output0: {
      cpuData: flat.reduce((acc, val, i) => { acc[i] = val; return acc; }, {}),
      dataLocation: "cpu",
      type: "float32",
      dims: outputTensor.dims,
      size: outputTensor.data.length
    }
  };
  fs.writeFileSync(outputRawJson, JSON.stringify(jsonRaw, null, 2));
  console.log(` Saída RAW salva: ${outputRawJson}`);

  // Processa e salva filtrado
  const [N, C, L] = outputTensor.dims;
  const data = outputTensor.data;
  const detections = [];

  for (let i = 0; i < L; i++) {
    const det = [];
    for (let j = 0; j < C; j++) {
      det.push(data[j * L + i]);
    }
    detections.push(det);
  }

const filtered = detections
  .filter(det => det[4] >= confThreshold)
  .map(det => {
    const [cx, cy, w, h] = det.slice(0, 4);
    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;
    return {
      bbox: [x1, y1, x2, y2],
      confidence: det[4],
      keypoints: Array.from({ length: 17 }, (_, i) => det.slice(5 + i * 3, 8 + i * 3))
    };
  });


  const jsonFiltered = {
    num_detections: filtered.length,
    detections: filtered
  };

  fs.writeFileSync(outputFilteredJson, JSON.stringify(jsonFiltered, null, 2));
  console.log(`Saída filtrada salva: ${outputFilteredJson}`);
}

main().catch(console.error);
