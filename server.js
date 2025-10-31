// server.js
const express = require('express');
const multer = require('multer');
const faceapi = require('face-api.js');
const tf = require('@tensorflow/tfjs');
const { MongoClient } = require('mongodb');
const canvas = require('canvas');
const path = require('path');
const fs = require('fs');
const cors = require('cors');


// === Setup Canvas bindings for face-api ===
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// === Express Setup ===
const app = express();
const upload = multer({ dest: 'uploads/' });
const PORT = 5000;
app.use(cors());
// === MongoDB Setup ===
const mongoURL = 'mongodb://127.0.0.1:27017';
const dbName = 'faceDB';
let db;
let embeddingsCache = []; // Keep embeddings in RAM for speed

// === Load Models ===
async function loadModels() {
  const modelPath = path.join(__dirname, 'models');
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  console.log('✅ FaceAPI models loaded');
}

// === Connect MongoDB ===
async function connectDB() {
  const client = new MongoClient(mongoURL);
  await client.connect();
  db = client.db(dbName);
  console.log('✅ Connected to MongoDB');
  await loadEmbeddingsToCache();
}

// === Preload all face embeddings to RAM ===
async function loadEmbeddingsToCache() {
  const faces = await db.collection('faces').find().toArray();
  embeddingsCache = faces.map(f => ({
    id: f._id,
    name: f.name,
    descriptor: new Float32Array(f.embedding),
  }));
  console.log(`📦 Loaded ${embeddingsCache.length} embeddings to memory`);
}

// === Image Analysis Helpers ===
function getAverageLuminance(imgData) {
  const data = imgData.data;
  let totalLum = 0;
  // Sample every 4th pixel for speed (still accurate enough)
  for (let i = 0; i < data.length; i += 16) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    // Standard luminance formula
    totalLum += (0.299 * r + 0.587 * g + 0.114 * b);
  }
  return totalLum / (data.length / 16);
}

function varianceOfLaplacian(imgData) {
  const width = imgData.width;
  const height = imgData.height;
  const data = imgData.data;
  let variance = 0;
  let mean = 0;
  const kernel = [-1, -1, -1, -1, 8, -1, -1, -1, -1]; // Laplacian kernel
  const values = [];

  // Convert to grayscale and apply Laplacian
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sum = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = ((y + ky) * width + (x + kx)) * 4;
          const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          sum += gray * kernel[(ky + 1) * 3 + (kx + 1)];
        }
      }
      values.push(sum);
      mean += sum;
    }
  }
  
  mean /= values.length;
  for (const v of values) {
    variance += (v - mean) ** 2;
  }
  return variance / values.length;
}

// === Analyze Face Image ===
async function analyzeFaceImage(imgPath) {
  const img = await canvas.loadImage(imgPath);
  
  // Detect all faces with landmarks and descriptors
  const detections = await faceapi
    .detectAllFaces(img)
    .withFaceLandmarks()
    .withFaceDescriptors();

  if (!detections || detections.length === 0) {
    return { status: 'reupload', reason: 'no-face-detected', message: 'No face detected - please provide a clear front-facing photo' };
  }

  if (detections.length > 1) {
    return { status: 'reupload', reason: 'multiple-faces', message: 'Multiple faces detected - please provide a photo with just one person' };
  }

  // Single face found - check quality
  const detection = detections[0];
  const box = detection.detection.box;
  const score = detection.detection.score;

  // Check face size
  if (box.height < 80 || box.width < 80) {
    return { status: 'reupload', reason: 'face-too-small', message: 'Face is too small - please move closer to the camera' };
  }

  // Check detection confidence
  if (score < 0.5) {
    return { status: 'reupload', reason: 'low-confidence', message: 'Face detection uncertain - please provide a clearer photo' };
  }

  // Crop face region and check image quality
  const faceCanvas = canvas.createCanvas(box.width, box.height);
  const ctx = faceCanvas.getContext('2d');
  ctx.drawImage(img, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
  const imgData = ctx.getImageData(0, 0, box.width, box.height);

  // Check brightness
  const lum = getAverageLuminance(imgData);
  if (lum < 40) {
    return { status: 'reupload', reason: 'too-dark', message: 'Image is too dark - please use better lighting' };
  }
  if (lum > 230) {
    return { status: 'reupload', reason: 'too-bright', message: 'Image is too bright - please reduce glare or bright lights' };
  }

  // Check blur
  const lapVar = varianceOfLaplacian(imgData);
  if (lapVar < 100) {
    return { status: 'reupload', reason: 'too-blurry', message: 'Image is blurry - please hold the camera steady' };
  }

  // All checks passed
  return { 
    status: 'ok',
    descriptor: detection.descriptor,
    score: score
  };
}

// === Compare Two Faces ===
function euclideanDistance(v1, v2) {
  return Math.sqrt(v1.reduce((sum, val, i) => sum + (val - v2[i]) ** 2, 0));
}

// === POST /verify ===
app.post('/verify', upload.single('image'), async (req, res) => {
  // Flow: analyze image quality first, then verify if quality is good
  const file = req.file;
  if (!file) return res.status(400).json({ status: 'error', message: 'No image uploaded' });

  const imgPath = file.path;
  try {
    // First analyze image quality
    const analysis = await analyzeFaceImage(imgPath);
    if (analysis.status === 'reupload') {
      return res.json(analysis); // Return quality issue with reason and message
    }

    // Image quality good, proceed with verification
    const embedding = analysis.descriptor;
    const distances = embeddingsCache.map(f => ({
      name: f.name,
      distance: euclideanDistance(f.descriptor, embedding),
    }));

    const matches = distances.filter(d => d.distance < 0.45); // 0.45 threshold
    if (matches.length === 1) {
      return res.json({ status: 'accept', name: matches[0].name, message: 'Accepted — good luck' });
    } else if (matches.length > 1) {
      return res.json({ status: 'reject', reason: 'multiple-matches', message: 'Rejected — multiple matches found' });
    } else {
      // No match found. If the client supplied a name in the same request,
      // auto-register the embedding. Otherwise instruct the client to provide
      // a name to register.
      const name = req.body && req.body.name;
      if (name) {
        const insertRes = await db.collection('faces').insertOne({ name, embedding: Array.from(embedding) });
        // append to in-memory cache without reloading everything
        embeddingsCache.push({ id: insertRes.insertedId, name, descriptor: new Float32Array(embedding) });
        return res.json({ status: 'registered', name, message: 'No previous match — registered successfully' });
      }
      return res.json({ status: 'register', message: 'No match found. Provide a `name` to register this face.' });
    }
  } catch (err) {
    console.error('Error in /verify:', err);
    return res.status(500).json({ status: 'error', message: String(err) });
  } finally {
    // cleanup temp file (best-effort)
    try { await fs.promises.unlink(imgPath); } catch (e) { /* ignore */ }
  }
});

// === POST /register ===
app.post('/register', upload.single('image'), async (req, res) => {
  const { name } = req.body;
  const file = req.file;
  if (!file) return res.status(400).json({ status: 'error', message: 'No image uploaded' });

  const imgPath = file.path;
  try {
    // First analyze image quality
    const analysis = await analyzeFaceImage(imgPath);
    if (analysis.status === 'reupload') {
      return res.json(analysis); // Return quality issue with reason and message
    }
    
    const embedding = analysis.descriptor;

    const insertRes = await db.collection('faces').insertOne({ name, embedding: Array.from(embedding) });
    // append to in-memory cache without reloading everything
    embeddingsCache.push({ id: insertRes.insertedId, name, descriptor: new Float32Array(embedding) });

    return res.json({ status: 'registered', name, message: 'Registered successfully' });
  } catch (err) {
    console.error('Error in /register:', err);
    return res.status(500).json({ status: 'error', message: String(err) });
  } finally {
    try { await fs.promises.unlink(imgPath); } catch (e) { /* ignore */ }
  }
});

// === GET /status ===
app.get('/status', (req, res) => {
  res.json({
    status: 'running',
    facesInMemory: embeddingsCache.length,
  });
});

// === Start Server ===
(async () => {
  await tf.ready();
  await tf.setBackend('cpu');
  await loadModels();
  await connectDB();
  app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));
})();
