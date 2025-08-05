# [flutter] Widget - FloatingActionButton: Floating Action Button

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`FloatingActionButton` adalah tombol aksi mengambang, yang biasanya digunakan untuk tindakan utama atau paling umum di suatu layar. Tombol ini sering ditempatkan di pojok kanan bawah layar dan menonjol secara visual.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/material/FloatingActionButton-class.html).

## Methods and Parameters
---
`FloatingActionButton` adalah properti dari `Scaffold`. Parameter yang umum digunakan adalah:
* **`onPressed`**: Wajib. Callback yang dipanggil saat tombol ditekan.
* **`child`**: Widget yang ditampilkan di dalam tombol, biasanya `Icon`.
* **`backgroundColor`** (opsional): Warna latar belakang tombol.
* **`tooltip`** (opsional): Teks yang akan ditampilkan saat pengguna menekan dan menahan tombol.
* **`mini`** (opsional): Jika `true`, tombol akan memiliki ukuran yang lebih kecil.

## Best Practices or Tips
---
* **Gunakan untuk Satu Aksi Utama**: Jangan gunakan `FloatingActionButton` untuk banyak aksi. Tombol ini harus mewakili satu aksi yang paling penting di layar tersebut (misalnya, "Buat Catatan Baru" atau "Tambahkan Item").
* **Gunakan `Hero` untuk Animasi**: Flutter memiliki animasi `Hero` bawaan untuk `FloatingActionButton` saat berpindah layar. Anda bisa memanfaatkan ini untuk membuat transisi yang halus.
* **Jangan Gunakan Tanpa Aksi**: Tombol ini harus selalu memiliki fungsi yang jelas. Jika tidak ada aksi utama, sebaiknya tidak menggunakan `FloatingActionButton`.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const FloatingActionButtonExample());

class FloatingActionButtonExample extends StatelessWidget {
  const FloatingActionButtonExample({Key? key}) : super(key: key);

  void _onPressed() {
    print('FloatingActionButton ditekan!');
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('FloatingActionButton Contoh')),
        body: const Center(
          child: Text('Tekan tombol mengambang di bawah'),
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: _onPressed,
          backgroundColor: Colors.blue,
          child: const Icon(Icons.add),
        ),
      ),
    );
  }
}