const express = require('express');
const multer = require('multer');
const uuid = require('uuid').v4;
const tf = require('@tensorflow/tfjs-node');

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
const app = express();
const upload = multer({ limits: { fileSize: 1000000 } }); // Batas ukuran file 1MB

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
    console.log('Input Tensor Shape:', tensor.shape);

    // Prediksi menggunakan model
    const predictions = model.predict(tensor).dataSync();
    console.log('Predictions:', predictions);

    // Interpretasi hasil
    const result = predictions[0] > 0.5 ? 'Cancer' : 'Non-cancer';
    const suggestion =
      result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Anda sehat!';
    const id = uuid();
    const createdAt = new Date().toISOString();

    // Kirim respons
    res.status(200).json({
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
    res.status(500).json({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi',
    });
  }
});

// Jalankan server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
