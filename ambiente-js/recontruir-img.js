const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');
const sharp = require('sharp');

async function plotKeypointsOnResizedImage(imagePath, jsonPath, outputPath, confThreshold = 0.1) {
  // Redimensiona a imagem para 640x640 com sharp
  const resizedBuffer = await sharp(imagePath).resize(640, 640).toBuffer();
  const img = await loadImage(resizedBuffer);

  // Cria canvas com 640x640
  const canvas = createCanvas(640, 640);
  const ctx = canvas.getContext('2d');

  // Desenha a imagem no canvas
  ctx.drawImage(img, 0, 0, 640, 640);

  
  const jsonData = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

  if (!jsonData.detections || jsonData.detections.length === 0) {
    throw new Error('Nenhuma detecção JSON.');
  }

  // Seleciona a (maior confiança dos 10)
  const bestDet = jsonData.detections.reduce((a, b) => a.confidence > b.confidence ? a : b);
  console.log(`Confiança: ${bestDet.confidence.toFixed(3)}`);

  // mostra os keypoints
  for (const [x, y, conf] of bestDet.keypoints) {
    if (conf >= confThreshold) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
    }
  }

  // Salva imagem final
  const outBuffer = canvas.toBuffer('image/png');
  fs.writeFileSync(outputPath, outBuffer);
  console.log(`Imagem com keypoints salva em: ${outputPath}`);
}


(async () => {
  try {
    await plotKeypointsOnResizedImage(
      path.resolve(__dirname, 'test-front-640.png'),
      path.resolve(__dirname, 'yolo_filtered.json'),
      path.resolve(__dirname, 'imagem_keypoints.png'),
      0.1
    );
  } catch (err) {
    console.error(err);
  }
})();
