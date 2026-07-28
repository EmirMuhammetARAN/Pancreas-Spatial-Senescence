import os
import subprocess
import sys

def main():
    # Güncellenen ve Merkezi Config ile çalışan dosyalarımızın listesi
    scripts = [
        "src/analysis/bolgeler_arasi_karsilastirma.py",
        "src/analysis/senescance_robustness.py",
        "src/analysis/bystander_analizi.py",
        "src/analysis/c_peptide_test.py",
        "src/analysis/check_beta_alpha_senescence.py",
        "src/analysis/clustering_analysis.py",
        "src/analysis/comprehensive_new_markers_test.py",
        "src/analysis/foxp3_macrophage_analizi.py",
        "src/analysis/hmgb1_analysis.py",
        "src/analysis/immune_infiltration.py"
    ]
    
    print("======================================================")
    print("MASTER ANALİZ ÇALIŞTIRICI BAŞLATILIYOR...")
    print(f"Toplam {len(scripts)} analiz sırayla çalıştırılacak.")
    print("======================================================\n")
    
    for i, script in enumerate(scripts, 1):
        print(f"[{i}/{len(scripts)}] Çalıştırılıyor: {script}")
        print("-" * 50)
        
        try:
            # subprocess ile scripti çalıştır (aktif terminalde logları göster)
            result = subprocess.run([sys.executable, script], check=True)
            print("-" * 50)
            print(f"✅ BAŞARILI: {script}\n")
        except subprocess.CalledProcessError as e:
            print("-" * 50)
            print(f"❌ HATA: {script} çalışırken bir hata oluştu (Exit Code: {e.returncode})\n")
            
    print("======================================================")
    print("BÜTÜN ANALİZLER TAMAMLANDI!")
    print("======================================================")

if __name__ == "__main__":
    main()
