# Projenin Amacı #

Bu proje, Türkiye genelinde 0.25° grid yapısı kullanılarak 2020–2025 yılları arasındaki orman yangılarını analiz eder gelecek yangın riskini tahmin etmeyi amaçlamaktadır.
Çalışma, çok faktörlü bir yapay zeka modeli ile geçmiş yangın verileri ve meteorolojik, arazi, insan etkisi gibi toplamda 40 parametreyi kullanmaktadır.

Amaç, hem önleyici erken uyarı sistemi geliştirmek hem de gelecekteki yangın risklerini görselleştirmek.

## 📊 Veri Kaynakları

| Kategori | Parametreler | Kaynak / Link |
|----------|--------------|----------------|
| 🔥 Yangın Geçmişi | Yangın lokasyonları, tarih, yoğunluk | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/country/) |
| 🌦️ Meteoroloji | Sıcaklık (min–max–ortalama), Nem, Rüzgâr hızı & yönü, Yağış, Buharlaşma, Güneşlenme süresi | [MGM](https://www.mgm.gov.tr/), [Open-Meteo API](https://open-meteo.com/en/docs/historical-weather-api) |
| 🌍 Arazi & Bitki Örtüsü | Arazi örtüsü sınıfları, orman yoğunluğu, toprak türü, eğim (slope), bakı (aspect) | [Copernicus Land Cover](https://land.copernicus.eu/pan-european/corine-land-cover), [Global Forest Change](https://earthenginepartners.appspot.com/science-2013-global-forest), [Copernicus DEM](https://registry.opendata.aws/copernicus-dem/), [Google Earth Engine Datasets](https://developers.google.com/earth-engine/datasets) |
| ⏳ Zaman Faktörleri | Gün, ay, mevsim, hafta içi/hafta sonu, resmî tatiller | [Mevzuat Takvimi](https://www.mevzuat.gov.tr/), Python `datetime` |
 
### 1. Yangın Geçmişi
NASA FIRMS (Fire Information for Resource Management System)
https://firms.modaps.eosdis.nasa.gov/country/
Geçmiş yıllardaki yangın lokasyonları (MODIS & VIIRS)
Yangın yoğunluğu ve tarih bilgileri

### 2. Meteoroloji Verileri
MGM (Meteoroloji Genel Müdürlüğü)
Open-Meteo Historical Weather API
http://open-meteo.com/en/docs/historical-weather-api
Sıcaklık (günlük min–max–ortalama)
Nem (%)
Rüzgar hızı (km/h) & rüzgar yönü (azimuth)
Yağış (mm)
Buharlaşma oranı
Güneşlenme süresi

### 3. Arazi & Bitki Örtüsü

Copernicus Land Cover (ESA CCI)
Google Earth Engine veri setleri
Orman yoğunluğu (% canopy cover)
Toprak türü
Eğim (slope)
Bakı (aspect – güneşe bakan yön)
Arazi kullanımı (orman, tarım, şehir vb.)

### 5. Zaman Faktörleri
Ay / gün / mevsim bilgisi
Hafta içi / hafta sonu

## Neden 0.25° Grid Yapısı?

Türkiye’nin enlem-boylam sınırları (25.6°E–45°E, 35.8°N–42.1°N) dikkate alındı.
0.25° (~27 km²) çözünürlük seçildi çünkü:
Meteoroloji ve uydu verilerinin çoğu bu çözünürlükte indirilebiliyor.
Daha yüksek çözünürlük (0.1° gibi) → veri boyutunu aşırı büyütüyor (~100 milyon satır).
Daha düşük çözünürlük (1° gibi) → yangın dağılımını fazla genelliyor.
Bu nedenle performans–doğruluk dengesi için 0.25° grid optimum tercih oldu.

### Modelleme Süreci

Veri Toplama → yıllık bazda tüm parametreler çekildi, CSV olarak kaydedildi.
Ön işleme → eksik veriler dolduruldu, normalize edildi.
Grid eşleştirme → her 0.25° hücre için parametreler birleştirildi.
Model →
Başlangıçta Logistic Regression, Random Forest ve XGBoost denendi.
Ardından zaman serisi öngörüsü için LSTM tabanlı model kuruldu.
Çıktı: her grid hücresine 0–100 arasında risk skoru.

### Güçlü Yönler

Çok faktörlü (meteoroloji + arazi + insan + zaman) yaklaşım.
Yüksek çözünürlüklü 0.25° grid ile yerel risk tahmini.
Tarihsel verilerle test edilip doğru ve yanlış tahminler gözlemlendi.
Görselleştirmeler: harita üzerinde geçmiş yangınlar (kırmızı), tahmini yangınlar (yeşil–sarı–kırmızı).

### Zayıf Yönler

Veriler farklı kaynaklardan geldiği için bazı bölgelerde eksik veya çelişkili olabilir.
Beta aşamasında: model doğruluğu tam garanti edilemez.
Günlük bazda anlık tahmin yapmak için daha güçlü altyapıya ihtiyaç var.
parametreler (ör. insan etkisi, turizm yoğunluğu) tahmini olarak modellendi.

### Gelecek Planları
2026 yazına kadar: 2020–2025 verilerinin tamamını GitHub’da paylaşmak.
Modeli canlı veri akışıyla besleyip gerçek zamanlı risk haritası oluşturmak.
Mobil uygulama entegrasyonu: vatandaşların bölgesel riskleri görebilmesi.
İtfaiye ve AFAD gibi kurumlarla işbirliği.

 
 ### Repo İçeriği

data/ → yıllık CSV verileri

# Algortima yapısı 

Modelin algoritmik yapısı da dikkat çekici bir şekilde tasarlandı. İlk aşamalarda basit sınıflandırma ve regresyon yöntemleriyle (Logistic Regression, Random Forest, XGBoost) farklı senaryolar test edildi. Ancak bu yöntemler yalnızca anlık ilişkilere odaklandığından, zaman serilerinde meydana gelen değişimleri yakalamakta yetersiz kaldı. Bu nedenle model, daha ileri bir yapay zeka yaklaşımı olan LSTM (Long Short-Term Memory) tabanlı sinir ağı ile geliştirildi. LSTM, geçmiş günlerin meteorolojik koşullarını ve arazi değişkenlerini dikkate alarak gelecekteki yangın riskini daha iyi tahmin edebiliyor. Bu yapı sayesinde model yalnızca “şu anda risk var mı?” sorusuna değil, aynı zamanda “önümüzdeki günlerde risk nasıl değişecek?” sorusuna da cevap verebiliyor.

Algoritma, her grid hücresi için 0 ile 100 arasında bir risk skoru üretiyor. Bu skorlar, görselleştirme aşamasında yeşilden kırmızıya uzanan bir renk skalasına dönüştürülüyor. Böylece düşük riskli bölgeler yeşil, orta riskli bölgeler sarı, yüksek riskli bölgeler ise kırmızı olarak harita üzerinde işaretleniyor. Bu görsel yaklaşım, karar vericilerin hızlıca riskli bölgeleri tespit etmesini sağlıyor.

Proje şu an hâlâ beta aşamasında. Doğru tahminler üretse de kimi zaman hatalı sonuçlar verebiliyor. Bunun nedeni hem veri kaynaklarının çeşitliliği hem de bazı parametrelerin henüz tam oturmamış olmasıdır. Ancak geliştirme süreci aktif olarak devam ediyor. 2026 yazına kadar, 2020–2025 yıllarına ait tüm veriler işlenmiş olacak ve tam algoritma kodları ile birlikte GitHub reposuna yüklenecek.Hedefim (Hayalim araştırmacılar, öğrenciler ve meraklılar modeli indirip kendi bilgisayarlarında çalıştırabilecek, farklı senaryoları test edebilecek bir program yaratmak.
<img width="1633" height="1305" alt="output" src="https://github.com/user-attachments/assets/2e14b193-9bff-4f59-b472-1d2015d5351b" />


# Sonuç görüntüsü 


