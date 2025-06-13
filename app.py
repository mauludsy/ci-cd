# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS agar frontend bisa berkomunikasi

# --- PENTING: PASTIKAN JALUR INI BENAR PADA SISTEM ANDA ---
model_path = "D:/deploy/KNeighborsClassifierModel.pkl"
scaler_path = "D:/deploy/Preprocessor.pkl"
# --- AKHIR PENTING ---

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Backend: Model dan preprocessor berhasil dimuat.")
except Exception as e:
    print(f"Backend Error: Gagal memuat model atau preprocessor: {e}")
    raise # Aplikasi akan berhenti jika gagal memuat model, ini disengaja agar tidak ada prediksi kosong

@app.route("/", methods=["GET"])
def home():
    return "API Prediksi Status Gizi Anak - KNN Model Berjalan!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"Backend: Menerima data dari frontend: {data}")

        nama = data.get("nama", "Tidak Diketahui")

        required_fields_for_model = ["umur", "tinggi_badan", "berat_badan"]
        for field in required_fields_for_model:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Field '{field}' wajib disertakan dan tidak boleh kosong."}), 400
            try:
                data[field] = float(data[field])
            except ValueError:
                return jsonify({"error": f"Nilai '{field}' harus berupa angka yang valid."}), 400

        umur = data["umur"]
        tinggi = data["tinggi_badan"]
        berat = data["berat_badan"]

        input_df = pd.DataFrame([{
            "Umur (bulan)": umur,
            "Tinggi Badan (cm)": tinggi,
            "Berat Badan (kg)": berat
        }])
        print(f"Backend: DataFrame input untuk scaling: {input_df}")

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        label_map = {0: "Normal", 1: "Stunted"}
        hasil_prediksi = label_map.get(prediction, "Status Tidak Diketahui")

        print(f"Backend: Hasil Prediksi untuk {nama}: {hasil_prediksi}")
        return jsonify({
            "nama": nama,
            "umur": umur,
            "tinggi_badan": tinggi,
            "berat_badan": berat,
            "prediksi_status_gizi": hasil_prediksi
        })

    except Exception as e:
        print(f"Backend Error: Terjadi kesalahan di endpoint /predict: {e}")
        return jsonify({"error": str(e), "message": "Terjadi kesalahan internal server. Mohon coba lagi."}), 500

if __name__ == "__main__":
    # --- PENTING: host='0.0.0.0' agar bisa diakses dari perangkat lain di jaringan lokal ---
    # Pastikan port 5000 tidak digunakan oleh aplikasi lain
    app.run(debug=True, host='0.0.0.0', port=5000)
    # --- AKHIR PENTING ---