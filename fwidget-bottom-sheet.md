# [flutter] Widget - BottomSheet: Action Sheet

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`BottomSheet` menampilkan lembar modal dari bagian bawah layar. Ini sering digunakan untuk menampilkan menu, opsi, atau informasi tambahan tanpa berpindah layar. Ada dua jenis utama: `showModalBottomSheet` (yang memblokir interaksi dengan widget lain) dan `showBottomSheet` (yang tidak memblokir).

Lihat dokumentasi resmi Flutter di [https://api.flutter.dev/flutter/material/showModalBottomSheet.html](https://api.flutter.dev/flutter/material/showModalBottomSheet.html).

## Methods and Parameters
---
Untuk menampilkan `BottomSheet`, Anda biasanya menggunakan fungsi **`showModalBottomSheet`**. Beberapa parameter penting yang sering digunakan adalah:

* **`context`**: Konteks build dari widget saat ini.
* **`builder`**: Fungsi yang mengembalikan widget yang akan ditampilkan di dalam sheet. Ini adalah parameter yang paling penting karena mendefinisikan konten UI Anda.
* **`isScrollControlled`** (opsional): Sebuah boolean. Jika diatur ke `true`, sheet dapat memenuhi hampir seluruh layar dan berguna saat konten di dalamnya dapat digulir.
* **`backgroundColor`** (opsional): Mengatur warna latar belakang untuk sheet.

Untuk menutup sheet, Anda menggunakan `Navigator.pop(context)`.

## Best Practices or Tips
---
* **Gunakan untuk Aksi Singkat**: `BottomSheet` ideal untuk tugas-tugas kecil yang tidak memerlukan navigasi penuh ke layar baru, seperti memilih opsi, konfirmasi cepat, atau menampilkan detail tambahan.
* **Kontrol Ukuran**: Bungkus konten `BottomSheet` Anda dengan `SizedBox` atau `Container` dan berikan tinggi tertentu agar tidak memenuhi seluruh layar secara default, kecuali jika Anda menggunakan `isScrollControlled: true`.
* **Pastikan Aman (SafeArea)**: Selalu bungkus konten `BottomSheet` Anda dengan widget `SafeArea` untuk memastikan konten tidak terpotong oleh notch, status bar, atau area gestur pada perangkat modern.
* **`DraggableScrollableSheet` untuk Fleksibilitas**: Jika Anda membutuhkan `BottomSheet` yang dapat ditarik dan diubah ukurannya oleh pengguna, pertimbangkan untuk menggunakan widget **`DraggableScrollableSheet`** sebagai gantinya.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const BottomSheetExample());

class BottomSheetExample extends StatelessWidget {
  const BottomSheetExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('BottomSheet Contoh'),
        ),
        body: Center(
          child: ElevatedButton(
            onPressed: () {
              showModalBottomSheet(
                context: context,
                isScrollControlled: true, // Sheet bisa digulir
                builder: (BuildContext context) {
                  return SafeArea(
                    child: Padding(
                      padding: EdgeInsets.only(
                        bottom: MediaQuery.of(context).viewInsets.bottom,
                      ),
                      child: SizedBox(
                        height: 300, // Menetapkan tinggi sheet
                        child: Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: <Widget>[
                              const Text('Ini adalah Modal Bottom Sheet'),
                              ElevatedButton(
                                child: const Text('Tutup'),
                                onPressed: () => Navigator.pop(context),
                              )
                            ],
                          ),
                        ),
                      ),
                    ),
                  );
                },
              );
            },
            child: const Text('Tampilkan Bottom Sheet'),
          ),
        ),
      ),
    );
  }
}