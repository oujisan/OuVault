# [flutter] Widget - PreferredSize: Custom Size

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)
## Description
---
Widget ini memungkinkan Anda untuk memberikan ukuran pilihan (`preferredSize`) kepada anak widgetnya. Ini sangat berguna ketika Anda ingin membuat `AppBar` dengan tinggi kustom.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/PreferredSize-class.html).

## Method and Function
---
`PreferredSize` pada dasarnya adalah wadah (wrapper) yang menerima satu parameter utama: `preferredSize`. Parameter ini membutuhkan objek `Size` yang mendefinisikan tinggi dan lebar yang diinginkan. Dalam banyak kasus, Anda akan menggunakannya dengan `Size.fromHeight()` untuk mengatur tinggi kustom pada `AppBar`.

**Contoh penggunaan utama:**
* **`AppBar` kustom**: Mengatur tinggi `AppBar` agar sesuai dengan kebutuhan desain, seperti menambahkan logo atau bilah pencarian di bawah judul.
* **Widget khusus lainnya**: Bisa digunakan untuk widget apa pun di mana Anda perlu menentukan ukuran preferensi agar induk (parent) dapat menatanya dengan benar.

## Best Practice atau Tips
---
* **Penggunaan Spesifik**: `PreferredSize` paling sering digunakan bersama `AppBar`. Jika Anda hanya ingin mengubah ukuran widget biasa, `SizedBox` atau `Container` mungkin merupakan pilihan yang lebih sederhana.
* **Responsif**: Gunakan `MediaQuery` atau `LayoutBuilder` di dalam `PreferredSize` jika Anda ingin ukuran widget bergantung pada ukuran layar perangkat.
* **Pertimbangkan `Flexible` atau `Expanded`**: Jika Anda menata widget di dalam `Row` atau `Column`, `Flexible` atau `Expanded` mungkin lebih sesuai untuk mengelola ruang, daripada secara statis menentukan ukuran dengan `PreferredSize`.

## Contoh
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const PreferredSizeExample());

class PreferredSizeExample extends StatelessWidget {
  const PreferredSizeExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: PreferredSize(
          preferredSize: const Size.fromHeight(100.0), // Tinggi kustom 100
          child: AppBar(
            title: const Text('AppBar Kustom'),
            backgroundColor: Colors.blueAccent,
            centerTitle: true,
          ),
        ),
        body: const Center(
          child: Text(
            'Ini adalah contoh PreferredSize untuk AppBar kustom.',
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }
}