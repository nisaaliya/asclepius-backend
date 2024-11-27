const express = require('express');
const multer = require('multer');
const uuid = require('uuid').v4;
const tf = require('@tensorflow/tfjs-node');
const { Firestore } = require('@google-cloud/firestore');
const admin = require('firebase-admin');
const serviceAccount = require('./path/to/serviceAccountKey.json'); // Sesuaikan path

// Inisialisasi admin SDK untuk Firestore
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

// Inisialisasi Firestore
const firestore = admin.firestore();

// Variabel global untuk menyimpan model
let model;

// Fungsi untuk memuat model dari Google Cloud Storage
const loadModel = async () => {
  const bucketName = 'bucketnisa27';
  const fileName = 'submissions-model/model.json';
  const modelURL = `https://storage.googleapis.com/${bucketName}/${fileName}`; // URL lengkap model
  try {
    console.log('Loading model from:', modelURL);
    model = await tf.loadGraphModel(modelURL); // Memuat model
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    process.exit(1); // Hentikan server jika model gagal dimuat
  }
};

// Fungsi untuk preprocessing gambar
const preprocessImage = (fileBuffer) => {
  const tensor = tf.node
    .decodeImage(fileBuffer) // Decode buffer gambar
    .resizeNearestNeighbor([224, 224]) // Ubah ukuran ke 224x224 (sesuai input model)
    .expandDims() // Tambahkan dimensi batch
    .toFloat(); // Ubah tipe data ke float32
  return tensor;
};

// Mulai memuat model saat server dijalankan
loadModel();

// Inisialisasi aplikasi Express
const app = express(); // <-- Ini bagian yang hilang

// Konfigurasi multer untuk menangani file upload dengan batas ukuran 1MB
const upload = multer({
  limits: { fileSize: 1000000 },
  fileFilter: (req, file, cb) => {
    // Filter file untuk memastikan hanya gambar yang diterima
    if (!file.mimetype.startsWith('image/')) {
      return cb(new Error('File is not an image'));
    }
    cb(null, true);
  },
});

// Endpoint prediksi
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    // Periksa apakah ada file
    const file = req.file;
    if (!file) {
      return res.status(400).json({
        status: 'fail',
        message: 'File not provided',
      });
    }

    // Preprocessing gambar
    const tensor = preprocessImage(file.buffer);

    // Prediksi menggunakan model
    const predictions = model.predict(tensor).dataSync();
    console.log('Predictions:', predictions);

    // Interpretasi hasil
    const result = predictions[0] > 0.5 ? 'Cancer' : 'Non-cancer';
    const suggestion =
      result === 'Cancer'
        ? 'Segera periksa ke dokter!'
        : 'Penyakit kanker tidak terdeteksi.';
    const id = uuid();
    const createdAt = new Date().toISOString();

    // Simpan data prediksi ke Firestore
    try {
      await firestore.collection('prediction').add({
        id,
        result,
        suggestion,
        createdAt,
      });
      console.log('Data saved to Firestore');
    } catch (firestoreError) {
      console.error('Error saving to Firestore:', firestoreError); // Log error Firestore
      return res.status(500).json({
        status: 'fail',
        message: 'Terjadi kesalahan saat menyimpan ke Firestore',
        error: firestoreError.message,
      });
    }

    // Kirim respons
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
    console.error('Error during prediction:', error); // Log error prediksi
    return res.status(400).json({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi',
      error: error.message,  // Tampilkan pesan error yang lebih jelas
    });
  }
});

// Endpoint untuk mengambil riwayat prediksi
app.get('/predict/histories', async (req, res) => {
    try {
      // Ambil semua dokumen dari koleksi `prediction`
      const snapshot = await firestore.collection('prediction').get();
  
      // Format data
      const histories = snapshot.docs.map((doc) => ({
        id: doc.id,
        history: doc.data(),
      }));
  
      // Kirim respons
      return res.status(200).json({
        status: 'success',
        data: histories,
      });
    } catch (error) {
      console.error('Error fetching prediction histories:', error);
      return res.status(500).json({
        status: 'fail',
        message: 'Terjadi kesalahan dalam mengambil riwayat prediksi',
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
