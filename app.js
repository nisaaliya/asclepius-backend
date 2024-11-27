const express = require('express');
const multer = require('multer');
const uuid = require('uuid').v4;
const { Storage } = require('@google-cloud/storage');
const { Firestore } = require('@google-cloud/firestore');
const tf = require('@tensorflow/tfjs-node');

// Initialize Firestore and Cloud Storage
const firestore = new Firestore();
const storage = new Storage();

// App setup
const app = express();
const upload = multer({ limits: { fileSize: 1000000 } });

app.use(express.json());

// Load model from Cloud Storage
let model;
const loadModel = async () => {
  const bucketName = 'bucketnisa27';
  const fileName = 'submissions-model/model.json';
  const modelURL = `gs://${bucketName}/${fileName}`;
  model = await tf.loadGraphModel(`https://storage.googleapis.com/${fileName}`);
};
loadModel();

// Prediction endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({
        status: 'fail',
        message: 'File not provided',
      });
    }

    const imageBuffer = file.buffer;
    const tensor = tf.node.decodeImage(imageBuffer)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const predictions = model.predict(tensor).dataSync();
    const result = predictions[0] > 0.5 ? 'Cancer' : 'Non-cancer';
    const suggestion =
      result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Anda sehat!';
    const id = uuid();
    const createdAt = new Date().toISOString();

    // Save to Firestore
    await firestore.collection('predictions').doc(id).set({
      id,
      result,
      suggestion,
      createdAt,
    });

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
    res.status(400).json({
      status: 'fail',
      message: 'Terjadi kesalahan dalam melakukan prediksi',
    });
  }
});

// Start server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
