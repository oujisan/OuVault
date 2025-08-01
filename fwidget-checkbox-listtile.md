# [flutter] Widget - CheckboxListTile: Checkbox List Item

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`CheckboxListTile` adalah kombinasi dari `ListTile` dan `Checkbox`. Ini adalah widget yang dirancang untuk menampilkan item daftar dengan kotak centang di dalamnya, ideal untuk daftar tugas atau opsi yang dapat dipilih.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/material/CheckboxListTile-class.html).

## Methods and Parameters
---
* **`value`**: Boolean yang menentukan apakah kotak centang dicentang (`true`) atau tidak (`false`).
* **`onChanged`**: Callback yang dipanggil setiap kali status kotak centang berubah. Callback ini menerima nilai boolean baru. Anda harus memperbarui status `value` di dalam `setState` di callback ini.
* **`title`** (opsional): Widget yang akan ditampilkan sebagai judul item daftar.
* **`subtitle`** (opsional): Widget yang akan ditampilkan di bawah judul.
* **`controlAffinity`** (opsional): Mengatur posisi kotak centang (`leading`, `trailing`, atau `platform`).

## Best Practices or Tips
---
* **Widget Berstatus**: `CheckboxListTile` adalah widget berstatus (`StatefulWidget`). Pastikan Anda menyimpan nilai `value` dalam variabel status dan memperbaruinya di dalam `onChanged` dengan `setState`.
* **Kemudahan Penggunaan**: Seluruh area `ListTile` responsif terhadap ketukan, yang berarti pengguna dapat mengetuk teks atau ikon untuk mencentang kotak, bukan hanya kotak centang itu sendiri.
* **Atur Warna**: Gunakan parameter `activeColor` untuk mengubah warna kotak centang saat dicentang, dan `checkColor` untuk mengubah warna tanda centangnya.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const CheckboxListTileExample());

class CheckboxListTileExample extends StatefulWidget {
  const CheckboxListTileExample({Key? key}) : super(key: key);

  @override
  State<CheckboxListTileExample> createState() => _CheckboxListTileExampleState();
}

class _CheckboxListTileExampleState extends State<CheckboxListTileExample> {
  bool _isChecked = false;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('CheckboxListTile Contoh')),
        body: Center(
          child: CheckboxListTile(
            title: const Text('Pilih Opsi Ini'),
            value: _isChecked,
            onChanged: (bool? newValue) {
              setState(() {
                _isChecked = newValue!;
              });
            },
            secondary: const Icon(Icons.info),
            activeColor: Colors.blue,
            checkColor: Colors.white,
          ),
        ),
      ),
    );
  }
}