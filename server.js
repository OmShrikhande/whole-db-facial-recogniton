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

// === Extract Face Embedding ===
async function getFaceEmbedding(imgPath) {
  const img = await canvas.loadImage(imgPath);
  const detections = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!detections) return null;
  return detections.descriptor;
}

// === Compare Two Faces ===
function euclideanDistance(v1, v2) {
  return Math.sqrt(v1.reduce((sum, val, i) => sum + (val - v2[i]) ** 2, 0));
}

// === POST /verify ===
app.post('/verify', upload.single('image'), async (req, res) => {
  // Flow: try to verify the uploaded image. If no match and a `name` is
  // provided in the same request, automatically register the face.
  const file = req.file;
  if (!file) return res.status(400).json({ status: 'error', message: 'No image uploaded' });

  const imgPath = file.path;
  let embedding;
  try {
    embedding = await getFaceEmbedding(imgPath);
    if (!embedding) return res.json({ status: 'no-face-detected', message: 'No face detected in the image' });

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
    const embedding = await getFaceEmbedding(imgPath);
    if (!embedding) return res.json({ status: 'no-face-detected', message: 'No face detected in the image' });

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
