import subprocess
import os

def run_script(script_path):
    print(f"\n==============================================")
    print(f"Menjalankan: {script_path}")
    print(f"==============================================")
    
    try:
        # Menjalankan script menggunakan python
        result = subprocess.run(['python', script_path], check=True, text=True)
        print(f"[OK] Selesai: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error saat menjalankan {script_path}")
        print(f"Exit code: {e.returncode}")
        exit(1)

def main():
    print("Memulai keseluruhan Pipeline Analisis Sentimen IHSG...\n")
    
    scripts = [
        "src/01_explorasi.py",
        "src/02_preprocessing.py",
        "src/03_sentiment.py",
        "src/04_agregasi_sentimen.py",
        "src/05_ihsg_trend.py",
        "src/06_merge_dan_korelasi.py",
        "src/07_visualisasi.py"
    ]
    
    # Pastikan current working directory benar
    cwd = os.getcwd()
    print(f"Current Directory: {cwd}")
    
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"[ERROR] Script tidak ditemukan: {script}")
            exit(1)
            
    print("\n[OK] PIPELINE SELESAI!")
    print("Silakan cek folder 'output/' untuk melihat hasil akhirnya.")

if __name__ == "__main__":
    main()
