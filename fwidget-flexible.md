# [flutter] Widget - Flexible: Flexible Child

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Flexible` adalah widget yang mengontrol bagaimana anak dari `Row`, `Column`, atau `Flex` mengisi ruang yang tersedia, tetapi dengan cara yang lebih fleksibel daripada `Expanded`. Berbeda dengan `Expanded`, `Flexible` memungkinkan anak untuk tidak mengisi seluruh ruang yang tersedia.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/Flexible-class.html).

## Methods and Parameters
---
`Flexible` memiliki parameter utama yang sama dengan `Expanded`:
* **`child`**: Widget yang akan disesuaikan ukurannya.
* **`flex`** (opsional): `int` yang menentukan proporsi ruang yang akan diisi. Nilai defaultnya adalah 1.
* **`fit`** (opsional): `FlexFit` yang menentukan cara anak mengisi ruang. Nilai defaultnya adalah `FlexFit.loose`.
    * **`FlexFit.loose`**: Anak dapat mengambil ruang sebesar yang dibutuhkannya, hingga `flex` yang ditentukan.
    * **`FlexFit.tight`**: Anak dipaksa untuk mengisi seluruh ruang yang tersedia, sama seperti `Expanded`.

## Best Practices or Tips
---
* **Pilih `Expanded` vs `Flexible`**: Gunakan `Expanded` ketika Anda ingin sebuah widget **harus** mengisi semua ruang kosong. Gunakan `Flexible` ketika Anda ingin widget tersebut **dapat** mengisi ruang, tetapi tidak harus.
* **Penggunaan Umum**: `Flexible` dengan `FlexFit.loose` sangat berguna saat Anda ingin widget menyesuaikan ukurannya berdasarkan kontennya, tetapi tetap bisa memenuhi sisa ruang jika kontennya kecil.
* **Pengendalian Lebih Baik**: Gunakan `Flexible` ketika Anda memerlukan kontrol lebih detail tentang bagaimana ruang didistribusikan di antara widget-widget di dalam `Row` atau `Column`.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const FlexibleExample());

class FlexibleExample extends StatelessWidget {
  const FlexibleExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Flexible Contoh')),
        body: Center(
          child: Column(
            children: <Widget>[
              // Kotak 1: Flexible dengan loose fit (ukuran sesuai konten)
              Flexible(
                child: Container(
                  color: Colors.red,
                  child: const Text('Flexible Loose Fit'),
                ),
              ),
              const SizedBox(height: 20),
              // Kotak 2: Flexible dengan tight fit (dipaksa mengisi ruang)
              Flexible(
                fit: FlexFit.tight,
                child: Container(
                  color: Colors.green,
                  child: const Center(child: Text('Flexible Tight Fit')),
                ),
              ),
              const SizedBox(height: 20),
              // Kotak 3: Expanded (sama dengan Flexible(fit: FlexFit.tight))
              Expanded(
                child: Container(
                  color: Colors.blue,
                  child: const Center(child: Text('Expanded')),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}