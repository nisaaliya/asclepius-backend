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
  