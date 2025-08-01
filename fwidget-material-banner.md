# [flutter] Widget - MaterialBanner: Notification Message

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Deskripsi
---
Widget ini menampilkan pesan non-intrusif di bagian atas layar. Widget ini berguna untuk memberikan informasi penting kepada pengguna tanpa mengganggu alur kerja mereka secara berlebihan, dan dapat ditutup.

Lihat dokumentasi resmi Flutter:;
 [https://api.flutter.dev/flutter/material/MaterialBanner-class.html](https://api.flutter.dev/flutter/material/MaterialBanner-class.html).

## Metode dan Fungsi
---
Untuk menampilkan `MaterialBanner`, Anda menggunakan metode `showMaterialBanner` dari `ScaffoldMessenger`. `ScaffoldMessenger` mengelola `MaterialBanner` dan `SnackBar` untuk `Scaffold` yang ada. Parameter utamanya adalah:
* **`content`**: Widget yang akan ditampilkan sebagai isi banner (misalnya, `Text`).
* **`actions`**: Daftar widget (`List<Widget>`) yang berfungsi sebagai tombol aksi di banner. Biasanya menggunakan `TextButton`.
* **`leading`** (opsional): Widget yang akan ditampilkan di awal banner, sering kali berupa `Icon`.

Untuk menutup banner, Anda menggunakan metode `hideCurrentMaterialBanner()` dari `ScaffoldMessenger` yang sama.

## Best Practice atau Tips
---
* **Gunakan untuk Info Non-Kritis**: `MaterialBanner` cocok untuk pesan seperti "Tidak ada koneksi internet" atau "Sinkronisasi selesai". Untuk error yang lebih penting yang membutuhkan respons langsung dari pengguna, pertimbangkan menggunakan `AlertDialog`.
* **Sertakan Aksi yang Jelas**: Pastikan tombol aksi (misalnya "TUTUP" atau "COBA LAGI") jelas dan fungsional. Ini membuat pengguna bisa berinteraksi dengan pesan tersebut.
* **Hindari Teks Terlalu Panjang**: Jaga agar pesan tetap singkat dan ringkas agar mudah dibaca dan tidak memakan banyak ruang di layar.

## Contoh
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const MaterialBannerExample());

class MaterialBannerExample extends StatelessWidget {
  const MaterialBannerExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('MaterialBanner Contoh'),
        ),
        body: Builder(
          builder: (context) => Center(
            child: ElevatedButton(
              onPressed: () {
                ScaffoldMessenger.of(context).showMaterialBanner(
                  MaterialBanner(
                    content: const Text('Ini adalah Material Banner.'),
                    leading: const Icon(Icons.info, color: Colors.white),
                    backgroundColor: Colors.blue,
                    actions: <Widget>[
                      TextButton(
                        onPressed: () {
                          ScaffoldMessenger.of(context)
                              .hideCurrentMaterialBanner();
                        },
                        child: const Text('TUTUP', style: TextStyle(color: Colors.white)),
                      ),
                    ],
                  ),
                );
              },
              child: const Text('Tampilkan Banner'),
            ),
          ),
        ),
      ),
    );
  }
}