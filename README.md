# Pancreatic Cellular Senescence Spatial Proteomics

Bu proje, bilgisayarlı görü (Computer Vision) ve PyTorch tabanlı Cellpose derin öğrenme modeli kullanılarak, insan pankreas dokusundaki hücresel yaşlanmanın (senesans) uzamsal proteomik verileri (PhenoCycler/CODEX) üzerinden incelenmesini amaçlamaktadır.

Proje, NIH SenNet veritabanından elde edilen 38 kanallı yüksek çözünürlüklü görüntüleri analiz ederek, senesansın anatomik bölgelere (Head, Body, Tail) göre nasıl değiştiğini tek hücre düzeyinde haritalandırmaktadır.

## Temel Özellikler (Methodology)

*   **Hücre Segmentasyonu:** PyTorch tabanlı `CellposeSAM v2` modeli kullanılarak DAPI ve E-cadherin kanalları üzerinden tam otomatik hücre sınırları tespiti.
*   **Büyük Veri İşleme:** Yüksek çözünürlüklü (örn. 28800x50400 piksel) QPTIFF görüntülerin `zarr` ve `tifffile` ile yönetilebilir 2048x2048 boyutlarında tile'lara bölünmesi.
*   **Uzamsal Analiz & Kümeleme:** Doku sınırlarını belirlemek için `DBSCAN` tabanlı kümeleme ve y-ekseni koordinatlarına dayalı anatomik bölge tespiti (Head, Tail).
*   **Özellik Çıkarımı:** Segmentasyonu yapılan yüz binlerce hücrenin 38 protein kanalındaki sinyal yoğunluklarının (Insulin, p16, Lamin B1, CD68 vb.) hesaplanması.
*   **Sağlam (Robust) Normalizasyon:** Outlier'lardan (aykırı değerler) etkilenmeyen, yaşa ve bölgeye özgü referans normalizasyon stratejisi.

## Kurulum ve Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olduğu bir Python sanal ortamı (virtual environment) gereklidir:

```bash
# Bağımlılıkları yükleyin
pip install numpy pandas tifffile zarr scikit-image scikit-learn matplotlib seaborn pyarrow fastparquet
# PyTorch ve Cellpose (CUDA desteği önerilir)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cellpose
```

## Proje Yapısı ve Kullanım (Pipeline)

Kodların çalışma sırası (Pipeline) aşağıdaki gibidir. Tüm süreç `process_pipeline.py` üzerinden entegre şekilde yürütülebilir.

1.  **`src/cutting/extract_all_tiles.py`**: QPTIFF formatındaki devasa doku görüntülerini Zarr ile okuyarak paralel işlenebilecek daha küçük `tiff` tile'lara böler.
2.  **`src/cutting/run_cellpose.py`**: Parçalanan tile'ları alır, PyTorch tabanlı CellposeSAM v2 modelini GPU üzerinde çalıştırarak her bir hücrenin segmentasyon maskesini (mask_tiff) çıkarır.
3.  **`src/cutting/extract_features.py`**: Çıkarılan maskeleri orijinal görüntülerle eşleştirir, 38 farklı kanal için her hücrenin ortalama sinyal yoğunluğunu (feature extraction) hesaplar ve veriyi `.parquet` formatında kaydeder.
4.  **`src/utils/filtering.py`**: Parquet dosyalarındaki hücresel veriyi alır, DBSCAN ile doku artefaktlarını temizler ve uzamsal (spatial) y-koordinatlarına göre pankreas bölgelerini ayırır.
5.  **`src/analysis/bolgeler_arasi_karsilastirma.py`**: Temizlenmiş ve bölgelere ayrılmış veriyi analiz eder, senesans (p16 artışı, Lamin B1 kaybı) oranlarını hesaplar ve grafikleri (boxplot, barplot) üretir.

## Tam Otomatik Pipeline Çalıştırma

Tüm adımları tek bir script üzerinden çalıştırmak için:

```bash
python src/cutting/process_pipeline.py
```

## Veri Seti

Kullanılan veriler NIH SenNet Consortium tarafından sağlanan Human Pancreas PhenoCycler/CODEX Atlas verileridir. Boyut sınırları nedeniyle ham `.qptiff` görüntüleri, üretilen `.parquet` veri dosyaları ve `.tiff` maskeleri bu depoya (repository) dahil edilmemiştir (Bkz. `.gitignore`).

## Lisans

Bu proje araştırma ve eğitim amaçlı açık kaynak olarak sunulmuştur.