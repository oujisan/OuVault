# [flutter] Widget - PopupMenuButton: Popup Menu

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`PopupMenuButton` adalah widget yang menampilkan menu pop-up (menu yang muncul di atas konten lain) saat ditekan. Ini sangat berguna untuk menampilkan daftar opsi yang tidak sering digunakan, seperti "edit", "hapus", atau "bagikan".

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/material/PopupMenuButton-class.html).

## Methods and Parameters
---
`PopupMenuButton` membutuhkan beberapa parameter untuk berfungsi:
* **`itemBuilder`**: Wajib. Fungsi ini mengembalikan daftar widget `PopupMenuItem` yang akan menjadi item-item di menu.
* **`onSelected`**: Callback yang dipanggil saat pengguna memilih salah satu item menu. Callback ini menerima nilai dari item yang dipilih.
* **`child`** (opsional): Widget yang berfungsi sebagai tombol pemicu menu. Jika tidak disediakan, Flutter akan menggunakan ikon `more_vert` secara default.
* **`icon`** (opsional): Widget ikon yang digunakan sebagai pemicu menu.

## Best Practices or Tips
---
* **Gunakan untuk Opsi Sekunder**: `PopupMenuButton` ideal untuk opsi yang tidak perlu ditampilkan di UI utama untuk menghemat ruang.
* **Nilai yang Jelas**: Setiap `PopupMenuItem` harus memiliki `value` yang unik. Ini adalah nilai yang akan diteruskan ke callback `onSelected` saat item dipilih.
* **Manajemen State**: Di dalam `onSelected`, Anda dapat menggunakan `setState` untuk memproses tindakan pengguna, seperti mengubah nilai atau menavigasi ke halaman lain.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const PopupMenuButtonExample());

enum MenuOptions { edit, delete, share }

class PopupMenuButtonExample extends StatelessWidget {
  const PopupMenuButtonExample({Key? key}) : super(key: key);

  void _onSelected(MenuOptions option) {
    switch (option) {
      case MenuOptions.edit:
        print('Edit dipilih');
        break;
      case MenuOptions.delete:
        print('Hapus dipilih');
        break;
      case MenuOptions.share:
        print('Bagikan dipilih');
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('PopupMenuButton Contoh'),
          actions: <Widget>[
            PopupMenuButton<MenuOptions>(
              onSelected: _onSelected,
              itemBuilder: (BuildContext context) {
                return <PopupMenuEntry<MenuOptions>>[
                  const PopupMenuItem<MenuOptions>(
                    value: MenuOptions.edit,
                    child: Text('Edit'),
                  ),
                  const PopupMenuItem<MenuOptions>(
                    value: MenuOptions.delete,
                    child: Text('Hapus'),
                  ),
                  const PopupMenuItem<MenuOptions>(
                    value: MenuOptions.share,
                    child: Text('Bagikan'),
                  ),
                ];
              },
            ),
          ],
        ),
        body: const Center(
          child: Text('Tekan ikon tiga titik di AppBar'),
        ),
      ),
    );
  }
}