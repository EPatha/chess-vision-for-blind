# chess-vision-for-blind

## Petunjuk singkat: annotasi papan 8x8 dan pemetaan kotak

Saya menambahkan utilitas sederhana untuk membantu anotasi papan catur dan pemetaan titik pixel ke kotak (algebraic: a1..h8).

File yang ditambahkan:
- `chess_board_utils.py` : fungsi utama untuk menghitung transformasi perspektif, memetakan pixel->square dan sebaliknya, menggambar overlay grid, dan mengonversi FEN (placement) menjadi peta bidak.
- `example_annotate.py`  : contoh skrip interaktif. Klik 4 sudut papan (TL,TR,BR,BL), kemudian skrip menggambar overlay dan (opsional) mengannotasi bidak jika diberikan FEN.
- `requirements.txt`    : dependensi (opencv-python, numpy).

Contoh penggunaan:
1. Pasang dependensi (disarankan virtualenv):

		pip install -r requirements.txt

2. Jalankan contoh (interaktif):

		python3 example_annotate.py --image path/to/your/photo.jpg --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

Instruksi singkat di jendela gambar:
- Klik 4 titik sudut papan dalam urutan TL,TR,BR,BL.
- Tekan 'r' untuk reset titik, 'q' bila selesai.

Asumsi orientasi:
- Implementasi sekarang menganggap bahwa setelah warp, titik (0,0) (kiri-atas) adalah kotak a8, dan (board_pixels, board_pixels) adalah h1.
	Dengan kata lain files kiri->kanan = a->h, ranks atas->bawah = 8->1.
	Jika kamera Anda menangkap papan dari sisi lain (mis. rotated), Anda dapat menukar urutan titik sudut atau memutar board yang diwarped.

Jika Anda ingin fitur tambahan (deteksi otomatis sudut papan, deteksi bidak otomatis, output urutan langkah, atau integrasi dengan engine), beri tahu saya dan saya akan tambahkan contoh dan/atau model deteksi.

## Dependencies (untuk Bab 2 - daftar paket dan kegunaan)

Ini proyek Python yang menggunakan beberapa paket utama. Berikut paket yang saya sertakan di `requirements.txt` beserta kegunaannya:

- numpy: operasi numerik dan array dasar.
- Pillow: pembacaan/penyimpanan gambar (dipakai oleh beberapa utilitas dan library).
- opencv-python: pemrosesan citra, pembacaan stream (cv2.VideoCapture), transformasi perspektif, dan drawing overlay.
- torch, torchvision, torchaudio: PyTorch — runtime inference yang dibutuhkan oleh beberapa model/versi `ultralytics`.
- ultralytics: paket YOLO (v8) untuk inferensi deteksi objek (mengunduh model seperti `yolov8n.pt`).
- Flask: web framework kecil untuk UI kalibrasi dan streaming MJPEG.

Catatan instalasi penting (macOS):
- Untuk PyTorch, penggunaan wheel yang tepat tergantung pada arsitektur (Intel vs Apple Silicon) dan apakah Anda ingin CUDA. Rekomendasi singkat untuk CPU-only macOS:

	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

- Jika `opencv-python` menimbulkan masalah pada build, coba pasang `opencv-python-headless` atau pastikan Xcode Command Line Tools terpasang:

	xcode-select --install

- `ultralytics` biasanya akan mengunduh model YOLO pertama kali saat dipakai; pastikan koneksi internet saat pertama inference.

Jika Anda ingin saya tambahkan file `environment.yml` (Conda) atau `Pipfile` untuk reproducibility, beri tahu arsitektur dan preferensi package manager (pip vs conda) dan saya buatkan.

just upload for github
add readme