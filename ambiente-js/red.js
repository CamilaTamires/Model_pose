const sharp = require('sharp');
const path = require('path');

const inputPath = path.resolve(__dirname, 'test-front.png');
const outputPath = path.resolve(__dirname, 'test-front-640.png');

sharp(inputPath)
  .resize(640, 640, {
    fit: 'contain', // mantém proporção e adiciona bordas se necessário
    background: { r: 0, g: 0, b: 0, alpha: 1 } 
  })
  .toFile(outputPath)
  .then(() => {
    console.log(' Imagem redimensionada :', outputPath);
  })
  .catch(err => {
    console.error('Erro:', err.message);
  });
