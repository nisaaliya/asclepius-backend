const express = require('express');
const multer = require('multer');
const uuid = require('uuid').v4;
const tf = require('@tensorflow/tfjs-node');
const { Firestore } = require('@google-cloud/firestore');

// Inisialisasi Firestore
const firestore = new Firestore();

// Variabel global untuk menyimpan model
let model;

// Fungsi untuk memuat model dari Google Cloud Storage
const loadModel = async () => {
  const bucketName = 'bucketnisa27';
  const fileName = 'submissions-model/model.json';
  const modelURL = `https://storage.googleapis.com/${bucketName}/${fileName}`;
  try {
    console.log('Loading model from:', modelURL);
    model = await tf.loadGraphModel(modelURL);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    process.exit(1);
  }
};

// Fungsi untuk preprocessing gambar
const preprocessImage = (fileBuffer) => {
  const tensor = tf.node
    .decodeImage(fileBuffer)
    .resizeNearestNeighbor([224, 224])
    .expandDims()
    .toFloat();
  return tensor;
};

// Mulai memuat model saat server dijalankan
loadModel();

// Inisialisasi aplikasi Express
const app = express();
const upload = multer({
  limits: { fileSize: 1000000 },
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith('image/')) {
      return cb(new Error('File is not an image'));
    }
    cb(null, true);
  },
});

// Endpoint prediksi
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({
        status: 'fail',
        message: 'File not provided',
      });
    }

    const tensor = preprocessImage(file.buffer);
    const predictions = model.predict(tensor).dataSync();
    console.log('Predictions:', predictions);

    const result = predictions[0] > 0.5 ? 'Cancer' : 'Non-cancer';
    const suggestion =
      result === 'Cancer'
        ? 'Segera periksa ke dokter!'
        : 'Penyakit kanker tidak terdeteksi.';
    const id = uuid();
    const createdAt = new Date().toISOString();

    // Simpan ke Firestore
    await firestore.collection('predictions').doc(id).set({
      id,
      result,
      suggestion,
      createdAt,
    });

    return res.status(200).json({
      status: 'success',
      message: 'Model is predicted successfully',
      data: {
        id,
        result,
        suggestion,
        createdAt,
      },
    });
  } catch (error) {
    console.error('Error during prediction:', error);
    return res.status(400).json({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi',
    });
  }
});

// Endpoint riwayat prediksi
app.get('/predict/histories', async (req, res) => {
  try {
    // Ambil semua dokumen dari collection "predictions"
    const snapshot = await firestore.collection('predictions').get();
    if (snapshot.empty) {
      return res.status(200).json({
        status: 'success',
        data: [],
      });
    }

    // Format data ke dalam array
    const histories = [];
    snapshot.forEach((doc) => {
      histories.push({
        id: doc.id,
        history: {
          ...doc.data(),
        },
      });
    });

    return res.status(200).json({
      status: 'success',
      data: histories,
    });
  } catch (error) {
    console.error('Error fetching histories:', error);
    return res.status(500).json({
      status: 'fail',
      message: 'Failed to fetch prediction histories',
    });
  }
});

// Middleware untuk menangani error payload terlalu besar (413)
app.use((err, req, res, next) => {
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({
      status: 'fail',
      message: 'Payload content length greater than maximum allowed: 1000000',
    });
  }
  if (err.message === 'File is not an image') {
    return res.status(400).json({
      status: 'fail',
      message: 'File is not a valid image',
    });
  }
  next(err);
});

// Jalankan server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
