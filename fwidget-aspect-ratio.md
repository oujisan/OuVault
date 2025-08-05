# [flutter] Widget - AspectRatio: Aspect Ratio Container

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`AspectRatio` adalah widget yang mencoba untuk menyesuaikan ukuran anaknya agar sesuai dengan rasio aspek (lebar dibagi tinggi) yang ditentukan, sambil tetap menghormati batasan yang diberikan oleh induknya. Ini sangat berguna untuk mempertahankan proporsi elemen seperti gambar atau video.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/AspectRatio-class.html).

## Methods and Parameters
---
`AspectRatio` memiliki dua parameter utama:
* **`aspectRatio`**: Wajib. `double` yang mewakili rasio lebar terhadap tinggi. Misalnya, 1.0 untuk rasio 1:1 (persegi), atau 16 / 9 untuk video widescreen.
* **`child`**: Widget yang ukurannya akan disesuaikan.

## Best Practices or Tips
---
* **Fleksibel**: `AspectRatio` akan menyesuaikan ukuran anaknya sejauh mungkin, tetapi tetap akan mematuhi batasan yang diberikan oleh widget induk (parent) di pohon widget.
* **Penggunaan Umum**: Ideal untuk memastikan gambar atau video tidak terdistorsi saat ukurannya berubah.
* **Kombinasikan dengan Widget Tata Letak**: Sering digunakan di dalam widget seperti `Column`, `Row`, atau `GridView` untuk mengontrol proporsi setiap item.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const AspectRatioExample());

class AspectRatioExample extends StatelessWidget {
  const AspectRatioExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('AspectRatio Contoh')),
        body: Center(
          child: Container(
            color: Colors.grey[200],
            width: double.infinity,
            height: 300,
            child: AspectRatio(
              aspectRatio: 16 / 9, // Rasio 16:9
              child: Container(
                color: Colors.blue,
                child: const Center(
                  child: Text(
                    'Rasio 16:9',
                    style: TextStyle(color: Colors.white, fontSize: 24),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}