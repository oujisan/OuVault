# [flutter] Widget - Expanded: Expand Child

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Expanded` adalah widget yang memperluas anaknya untuk mengisi ruang kosong yang tersedia di dalam `Row`, `Column`, atau `Flex`. Ini adalah widget tata letak yang sangat penting untuk membuat antarmuka yang responsif dan mengisi ruang secara merata.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/Expanded-class.html).

## Methods and Parameters
---
`Expanded` memiliki dua parameter utama:
* **`child`**: Widget yang akan diperluas.
* **`flex`** (opsional): `int` yang menentukan seberapa besar ruang yang harus diisi oleh widget ini dibandingkan dengan widget `Expanded` lainnya. Nilai defaultnya adalah 1.

## Best Practices or Tips
---
* **Gunakan di dalam `Row` atau `Column`**: `Expanded` hanya berfungsi jika digunakan sebagai anak langsung dari `Row`, `Column`, atau `Flex`.
* **Gunakan untuk Mengisi Ruang**: Gunakan `Expanded` ketika Anda ingin sebuah widget mengambil semua ruang sisa yang tersedia setelah widget lain di dalam `Row` atau `Column` telah diatur ukurannya.
* **Kontrol dengan `flex`**: Gunakan `flex` untuk mengontrol pembagian ruang. Misalnya, jika Anda memiliki dua widget `Expanded`, satu dengan `flex: 1` dan yang lainnya dengan `flex: 2`, maka widget kedua akan mengambil dua kali lebih banyak ruang daripada widget pertama.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const ExpandedExample());

class ExpandedExample extends StatelessWidget {
  const ExpandedExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Expanded Contoh')),
        body: Center(
          child: Column(
            children: <Widget>[
              // Kotak 1: Ukuran tetap
              Container(
                height: 50,
                color: Colors.red,
              ),
              // Kotak 2: Mengisi sisa ruang
              Expanded(
                child: Container(
                  color: Colors.green,
                  child: const Center(child: Text('Expanded (flex: 1)')),
                ),
              ),
              // Kotak 3: Juga mengisi sisa ruang
              Expanded(
                flex: 2,
                child: Container(
                  color: Colors.blue,
                  child: const Center(child: Text('Expanded (flex: 2)')),
                ),
              ),
              // Kotak 4: Ukuran tetap
              Container(
                height: 50,
                color: Colors.yellow,
              ),
            ],
          ),
        ),
      ),
    );
  }
}