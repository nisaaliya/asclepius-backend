# Menggunakan image dasar Node.js
FROM node:18

# Menentukan direktori kerja dalam container
WORKDIR /app

# Menyalin package.json dan package-lock.json
COPY package*.json ./

# Menginstal dependensi aplikasi
RUN npm install

# Menyalin seluruh kode aplikasi ke dalam container
COPY . .

# Mengekspos port yang digunakan oleh aplikasi
EXPOSE 8080

# Menjalankan aplikasi
CMD ["node", "app.js"]
