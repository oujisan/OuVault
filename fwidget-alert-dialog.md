# [flutter] Widget - AlertDialog: Pop-up Alert

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`AlertDialog` adalah widget yang menampilkan dialog pop-up yang memberi tahu pengguna tentang suatu situasi dan dapat meminta mereka untuk membuat keputusan. Ini biasanya digunakan untuk pesan peringatan atau konfirmasi penting.

Lihat dokumentasi resmi Flutter di [https://api.flutter.dev/flutter/material/AlertDialog-class.html](https://api.flutter.dev/flutter/material/AlertDialog-class.html).

## Methods and Parameters
---
Anda menggunakan fungsi **`showDialog`** untuk menampilkan `AlertDialog`. Parameter utamanya adalah:
* **`context`**: Konteks build dari widget saat ini.
* **`builder`**: Fungsi yang mengembalikan `AlertDialog`.

Parameter `AlertDialog` itu sendiri:
* **`title`** (opsional): Widget untuk judul dialog.
* **`content`** (opsional): Widget untuk isi dialog, biasanya `Text`.
* **`actions`** (opsional): Daftar widget (`List<Widget>`) yang berfungsi sebagai tombol aksi di bagian bawah dialog.

## Best Practices or Tips
---
* **Blokir Interaksi**: `AlertDialog` memblokir interaksi pengguna dengan UI di belakangnya. Gunakan ini untuk hal-hal yang benar-benar memerlukan perhatian pengguna.
* **Gunakan `Navigator.pop()`**: Untuk menutup dialog, Anda harus memanggil `Navigator.pop(context)` di dalam callback tombol aksi.
* **Tindakan yang Jelas**: Pastikan teks di tombol aksi jelas (misalnya "Hapus", "Batalkan") dan memandu pengguna untuk membuat keputusan yang tepat.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const AlertDialogExample());

class AlertDialogExample extends StatelessWidget {
  const AlertDialogExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('AlertDialog Contoh')),
        body: Center(
          child: ElevatedButton(
            onPressed: () {
              showDialog(
                context: context,
                builder: (BuildContext context) {
                  return AlertDialog(
                    title: const Text('Konfirmasi'),
                    content: const Text('Apakah Anda yakin ingin melanjutkan?'),
                    actions: <Widget>[
                      TextButton(
                        onPressed: () {
                          Navigator.of(context).pop(); // Tutup dialog
                        },
                        child: const Text('Batalkan'),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.of(context).pop(); // Tutup dialog
                          print('Aksi dikonfirmasi!');
                        },
                        child: const Text('Ya'),
                      ),
                    ],
                  );
                },
              );
            },
            child: const Text('Tampilkan Dialog'),
          ),
        ),
      ),
    );
  }
}