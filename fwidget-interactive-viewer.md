# [flutter] Widget - InteractiveViewer: Pan and Zoom

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`InteractiveViewer` memungkinkan pengguna untuk memperbesar (zoom), menggeser (pan), atau memutar (rotate) widget anaknya. Ini sangat berguna untuk menampilkan gambar, peta, atau diagram yang lebih besar dari layar.

Lihat dokumentasi resmi Flutter di [https://api.flutter.dev/flutter/widgets/InteractiveViewer-class.html](https://api.flutter.dev/flutter/widgets/InteractiveViewer-class.html).

## Methods and Parameters
---
`InteractiveViewer` memiliki beberapa parameter untuk mengontrol interaktivitasnya:
* **`child`**: Widget yang akan diperbesar dan digeser.
* **`minScale`**: Faktor skala minimal yang diizinkan (default: 0.25).
* **`maxScale`**: Faktor skala maksimal yang diizinkan (default: 2.5).
* **`boundaryMargin`**: Jarak margin yang menentukan seberapa jauh pengguna dapat menggeser di luar batas konten.
* **`panEnabled`**: Jika `false`, menggeser (panning) dinonaktifkan.
* **`scaleEnabled`**: Jika `false`, memperbesar (scaling) dinonaktifkan.

## Best Practices or Tips
---
* **Batasi Skala**: Atur `minScale` dan `maxScale` ke nilai yang wajar untuk mencegah pengguna memperbesar terlalu jauh, yang dapat menyebabkan masalah performa atau pengalaman pengguna yang buruk.
* **Gabungkan dengan Widget Lain**: Gunakan `InteractiveViewer` untuk membungkus widget seperti `Image.network` atau `SingleChildScrollView` untuk membuat konten interaktif.
* **Penggunaan Memori**: Perhatikan penggunaan memori saat memperbesar gambar beresolusi sangat tinggi, karena ini dapat membebani perangkat.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const InteractiveViewerExample());

class InteractiveViewerExample extends StatelessWidget {
  const InteractiveViewerExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('InteractiveViewer Contoh')),
        body: InteractiveViewer(
          boundaryMargin: const EdgeInsets.all(20.0),
          minScale: 0.5,
          maxScale: 4.0,
          child: Center(
            child: Image.network(
              'https://picsum.photos/600/600',
              fit: BoxFit.cover,
            ),
          ),
        ),
      ),
    );
  }
}