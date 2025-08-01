# [flutter] Widget - CupertinoContextMenu: iOS Context Menu

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`CupertinoContextMenu` adalah widget yang menampilkan menu konteks gaya iOS saat widget anaknya ditekan dan ditahan (long press). Menu ini muncul dengan efek animasi yang halus.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/cupertino/CupertinoContextMenu-class.html).

## Methods and Parameters
---
* **`child`**: Widget yang akan menampilkan menu konteks saat ditekan lama.
* **`actions`**: Daftar widget `CupertinoContextMenuAction`. Ini adalah item-item yang akan muncul di menu. Setiap `CupertinoContextMenuAction` memiliki `child` dan callback `onPressed`.
* **`preview`**: Widget yang ditampilkan dalam bentuk pratinjau yang membesar saat menu ditekan lama. Biasanya sama dengan `child` atau versi yang sedikit diubah.

## Best Practices or Tips
---
* **Penggunaan Konteks**: `CupertinoContextMenu` cocok untuk memberikan opsi tambahan terkait sebuah item, seperti "Edit", "Hapus", atau "Bagikan", tanpa memakan ruang di antarmuka utama.
* **Interaksi yang Jelas**: Pastikan pengguna tahu bahwa sebuah item dapat menampilkan menu konteks. Anda dapat menggunakan indikator visual atau teks bantuan untuk menunjukkan interaksi ini.
* **Gunakan untuk Platform iOS**: Karena ini adalah widget gaya Cupertino, disarankan untuk menggunakannya secara eksklusif saat menargetkan platform iOS untuk konsistensi visual.

## Example
---
```dart
import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';

void main() => runApp(const CupertinoContextMenuExample());

class CupertinoContextMenuExample extends StatelessWidget {
  const CupertinoContextMenuExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('CupertinoContextMenu Contoh')),
        body: Center(
          child: SizedBox(
            width: 200,
            height: 200,
            child: CupertinoContextMenu(
              actions: <Widget>[
                CupertinoContextMenuAction(
                  child: const Text('Aksi 1'),
                  onPressed: () {
                    Navigator.pop(context);
                    print('Aksi 1 dipanggil!');
                  },
                ),
                CupertinoContextMenuAction(
                  child: const Text('Hapus'),
                  isDestructiveAction: true,
                  onPressed: () {
                    Navigator.pop(context);
                    print('Hapus dipanggil!');
                  },
                ),
              ],
              child: Container(
                color: Colors.blue,
                child: const Center(
                  child: Text('Tekan dan Tahan', style: TextStyle(color: Colors.white)),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}