### Projenin Amacı ###

Bu proje, Türkiye genelinde 0.25° grid yapısı kullanılarak 2020–2025 yılları arasındaki orman yangılarını analiz eder gelecek yangın riskini tahmin etmeyi amaçlamaktadır.
Çalışma, çok faktörlü bir yapay zeka modeli ile geçmiş yangın verileri ve meteorolojik, arazi, insan etkisi gibi toplamda 40 parametreyi kullanmaktadır.

Amaç, hem önleyici erken uyarı sistemi geliştirmek hem de gelecekteki yangın risklerini görselleştirmek.


# 1. Yangın Geçmişi
NASA FIRMS (Fire Information for Resource Management System)
https://firms.modaps.eosdis.nasa.gov/country/
Geçmiş yıllardaki yangın lokasyonları (MODIS & VIIRS)
Yangın yoğunluğu ve tarih bilgileri

# 2. Meteoroloji Verileri
MGM (Meteoroloji Genel Müdürlüğü)
Open-Meteo Historical Weather API
http://open-meteo.com/en/docs/historical-weather-api
Sıcaklık (günlük min–max–ortalama)
Nem (%)
Rüzgar hızı (km/h) & rüzgar yönü (azimuth)
Yağış (mm)
Buharlaşma oranı
Güneşlenme süresi

# 3. Arazi & Bitki Örtüsü

Copernicus Land Cover (ESA CCI)
Google Earth Engine veri setleri
Orman yoğunluğu (% canopy cover)
Toprak türü
Eğim (slope)
Bakı (aspect – güneşe bakan yön)
Arazi kullanımı (orman, tarım, şehir vb.)

# 5. Zaman Faktörleri
Ay / gün / mevsim bilgisi
Hafta içi / hafta sonu

### Neden 0.25° Grid Yapısı?

Türkiye’nin enlem-boylam sınırları (25.6°E–45°E, 35.8°N–42.1°N) dikkate alındı.
0.25° (~27 km²) çözünürlük seçildi çünkü:
Meteoroloji ve uydu verilerinin çoğu bu çözünürlükte indirilebiliyor.
Daha yüksek çözünürlük (0.1° gibi) → veri boyutunu aşırı büyütüyor (~100 milyon satır).
Daha düşük çözünürlük (1° gibi) → yangın dağılımını fazla genelliyor.
Bu nedenle performans–doğruluk dengesi için 0.25° grid optimum tercih oldu.

# Modelleme Süreci

Veri Toplama → yıllık bazda tüm parametreler çekildi, CSV olarak kaydedildi.
Ön işleme → eksik veriler dolduruldu, normalize edildi.
Grid eşleştirme → her 0.25° hücre için parametreler birleştirildi.
Model →
Başlangıçta Logistic Regression, Random Forest ve XGBoost denendi.
Ardından zaman serisi öngörüsü için LSTM tabanlı model kuruldu.
Çıktı: her grid hücresine 0–100 arasında risk skoru.

# Güçlü Yönler

Çok faktörlü (meteoroloji + arazi + insan + zaman) yaklaşım.
Yüksek çözünürlüklü 0.25° grid ile yerel risk tahmini.
Tarihsel verilerle test edilip doğru ve yanlış tahminler gözlemlendi.
Görselleştirmeler: harita üzerinde geçmiş yangınlar (kırmızı), tahmini yangınlar (yeşil–sarı–kırmızı).

# Zayıf Yönler

Veriler farklı kaynaklardan geldiği için bazı bölgelerde eksik veya çelişkili olabilir.
Beta aşamasında: model doğruluğu tam garanti edilemez.
Günlük bazda anlık tahmin yapmak için daha güçlü altyapıya ihtiyaç var.
parametreler (ör. insan etkisi, turizm yoğunluğu) tahmini olarak modellendi.

# Gelecek Planları
2026 yazına kadar: 2020–2025 verilerinin tamamını GitHub’da paylaşmak.
Modeli canlı veri akışıyla besleyip gerçek zamanlı risk haritası oluşturmak.
Mobil uygulama entegrasyonu: vatandaşların bölgesel riskleri görebilmesi.
İtfaiye ve AFAD gibi kurumlarla işbirliği.

 
 # Repo İçeriği

data/ → yıllık CSV verileri






