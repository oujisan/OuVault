# [flutter] Widget - Card: Card Container

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Card` adalah wadah yang menampilkan informasi terkait dalam satu blok. Ini memiliki sudut membulat dan bayangan (elevasi) yang memberikan efek 3D, membuatnya menonjol dari latar belakang.

Lihat dokumentasi resmi Flutter di [https://api.flutter.dev/flutter/material/Card-class.html](https://api.flutter.dev/flutter/material/Card-class.html).

## Methods and Parameters
---
`Card` memiliki beberapa parameter untuk mengustomisasi tampilannya:
* **`child`**: Widget yang berada di dalam card.
* **`elevation`**: `double` yang menentukan ketinggian card, yang memengaruhi ukuran bayangannya.
* **`color`**: Warna latar belakang card.
* **`margin`**: Jarak margin di sekitar card.
* **`shape`**: Mengatur bentuk card. Secara default, ini adalah `RoundedRectangleBorder`.

## Best Practices or Tips
---
* **Gunakan untuk Pengelompokan**: `Card` ideal untuk mengelompokkan konten yang berhubungan, seperti detail produk, informasi kontak, atau entri blog, agar mudah dibaca oleh pengguna.
* **Jangan Gunakan di Dalam `Card` Lain**: Hindari menumpuk `Card` di dalam `Card` lain. Ini dapat menyebabkan tumpang tindih bayangan yang tidak diinginkan.
* **Interaktivitas**: Jika Anda ingin `Card` dapat diketuk, bungkusnya dengan `InkWell` atau `GestureDetector`, atau gunakan widget `ListTile` di dalamnya yang sudah memiliki interaktivitas bawaan.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const CardExample());

class CardExample extends StatelessWidget {
  const CardExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Card Contoh')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Card(
              elevation: 5,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(15.0),
              ),
              child: const Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    ListTile(
                      leading: Icon(Icons.album),
                      title: Text('The Enchanted Nightingale'),
                      subtitle: Text('Music by Jupiter'),
                    ),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: <Widget>[
                        TextButton(
                          onPressed: null,
                          child: Text('LISTEN'),
                        ),
                        SizedBox(width: 8),
                        TextButton(
                          onPressed: null,
                          child: Text('BUY'),
                        ),
                        SizedBox(width: 8),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}