// server.js
require('dotenv').config();
const express = require('express');
const multer = require('multer');
const faceapi = require('face-api.js');
const tf = require('@tensorflow/tfjs-node');
const { MongoClient } = require('mongodb');
const canvas = require('canvas');
const path = require('path');
const cors = require('cors');


// === Setup Canvas bindings for face-api ===
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// === Express Setup ===
const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 5 * 1024 * 1024 } });
const PORT = 5000;
app.use(cors());
// === MongoDB Setup ===
const mongoURL = process.env.mongouri;
const dbName = process.env.dbName;
let db;
let embeddingsCache = []; // Keep embeddings in RAM for speed
let registeredEmbeddingsCache = [];

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
  const client = new MongoClient(mongoURL, {
    tls: true,
    tlsAllowInvalidCertificates: false,
    serverSelectionTimeoutMS: 5000
  });
  await client.connect();
  db = client.db(dbName);
  
  // Create compound index on usercode and status
  // This allows duplicate usercodes only if they have different statuses
  await db.collection('image_verifications').createIndex(
    { usercode: 1, status: 1 },
    { 
      unique: true,
      partialFilterExpression: { status: "registered" } // Only enforce uniqueness for registered entries
    }
  );
  
  // Create indexes for faster face similarity searches
  await db.collection('image_verifications').createIndex(
    { imagetext: 1 },
    { name: "face_embedding_index" }
  );
  
  // Create index for querying rejected attempts
  await db.collection('image_verifications').createIndex(
    { status: 1, createdAt: -1 },
    { name: "status_time_index" }
  );

  console.log('✅ Connected to MongoDB and created indexes');
  await loadEmbeddingsToCache();
}

// === Preload all face embeddings to RAM ===
async function loadEmbeddingsToCache() {
  const faces = await db.collection('faces').find().project({ name: 1, embedding: 1 }).toArray();
  embeddingsCache = faces
    .filter(f => Array.isArray(f.embedding))
    .map(f => ({
      id: f._id,
      name: f.name,
      descriptor: Float32Array.from(f.embedding),
    }));

  const registered = await db.collection('image_verifications')
    .find({ status: 'registered' })
    .project({ usercode: 1, imagetext: 1 })
    .toArray();

  registeredEmbeddingsCache = registered
    .filter(r => Array.isArray(r.imagetext))
    .map(r => ({
      id: r._id,
      usercode: r.usercode,
      descriptor: Float32Array.from(r.imagetext),
    }));

  console.log(`📦 Loaded ${embeddingsCache.length} reference embeddings and ${registeredEmbeddingsCache.length} registered embeddings to memory`);
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

  // Convert to grayscale and apply Laplacian (sample every 2nd pixel for speed)
  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
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
async function analyzeFaceImage(source) {
  let img = await canvas.loadImage(source);

  // Resize image if too large to speed up processing
  const maxDimension = 800;
  if (img.width > maxDimension || img.height > maxDimension) {
    const scale = Math.min(maxDimension / img.width, maxDimension / img.height);
    const newWidth = Math.floor(img.width * scale);
    const newHeight = Math.floor(img.height * scale);
    const resizedCanvas = canvas.createCanvas(newWidth, newHeight);
    const ctx = resizedCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0, newWidth, newHeight);
    img = resizedCanvas;
  }

  // Detect all faces with descriptors (landmarks not needed)
  const detections = await faceapi
    .detectAllFaces(img)
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
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    const diff = v1[i] - v2[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

// === Check for duplicate faces in database ===
async function checkDuplicateFace(embedding, threshold = 0.45) {
  let bestMatch = null;
  for (const face of registeredEmbeddingsCache) {
    const distance = euclideanDistance(embedding, face.descriptor);
    if (distance < threshold && (!bestMatch || distance < bestMatch.distance)) {
      bestMatch = {
        usercode: face.usercode,
        distance,
      };
    }
  }

  if (bestMatch) {
    return {
      isDuplicate: true,
      existingUsercode: bestMatch.usercode,
      distance: bestMatch.distance,
    };
  }

  return { isDuplicate: false };
}

// === POST /verify ===
app.post('/verify', upload.single('image'), async (req, res) => {
  // Flow: analyze image quality first, then verify if quality is good
  const file = req.file;
  if (!file || !file.buffer) return res.status(400).json({ status: 'error', message: 'No image uploaded' });

  try {
    const analysis = await analyzeFaceImage(file.buffer);
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
      // No match found - validate and store the new face
      const { usercode } = req.body;
      
      // Validate usercode
      if (!usercode || typeof usercode !== 'string' || usercode.trim().length === 0) {
        return res.json({
          status: 'error',
          message: 'A valid usercode is required for verification'
        });
      }

      // Check if usercode already exists
      const existingUser = await db.collection('image_verifications').findOne({ usercode: usercode });
      if (existingUser) {
        return res.json({
          status: 'error',
          message: 'This usercode is already registered. Please use a different code.'
        });
      }

      // Check if this face exists for another usercode
      const duplicateCheck = await checkDuplicateFace(embedding);
      if (duplicateCheck.isDuplicate) {
        // Store the rejected attempt in the database
        try {
          await db.collection('image_verifications').insertOne({
            usercode: usercode.trim(),
            imagetext: Array.from(embedding),
            status: 'rejected',
            rejectionReason: 'duplicate-face',
            matchedUsercode: duplicateCheck.existingUsercode,
            similarity: Math.round((1 - duplicateCheck.distance) * 100),
            createdAt: new Date()
          });
          console.log('✅ Stored rejected attempt for duplicate face');
        } catch (err) {
          console.error('Failed to store rejected attempt:', err);
        }

        return res.json({
          status: 'duplicate-face',
          message: 'This face is already registered with usercode:',
          existingUsercode: duplicateCheck.existingUsercode,
          similarity: Math.round((1 - duplicateCheck.distance) * 100) + '%'
        });
      }
      
      // All checks passed - store in image_verifications collection
      try {
        const storeResult = await db.collection('image_verifications').insertOne({
          usercode: usercode.trim(),
          imagetext: Array.from(embedding),
          status: 'registered',
          createdAt: new Date()
        });

        if (storeResult && storeResult.insertedId) {
          console.log('✅ Stored new face embedding:', storeResult.insertedId.toString());
          return res.json({ 
            status: 'registered',
            message: 'Face registered successfully for verification',
            notification: 'Your face has been registered for future verification'
          });
        }
      } catch (dbError) {
        if (dbError.code === 11000) { // MongoDB duplicate key error
          return res.json({
            status: 'error',
            message: 'This usercode is already registered. Please use a different code.'
          });
        }
        throw dbError; // Let the main error handler catch other DB errors
      }
      
      return res.json({ 
        status: 'error',
        message: 'Failed to store face data'
      });
    }
  } catch (err) {
    console.error('Error in /verify:', err);
    return res.status(500).json({ status: 'error', message: String(err) });
  } finally {
    // cleanup temp file (best-effort)
    try { await fs.promises.unlink(imgPath); } catch (e) { /* ignore */ }
  }
});


// === Start Server ===
(async () => {
  await tf.ready();
  await tf.setBackend('cpu');
  await loadModels();
  await connectDB();
  app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));
})();