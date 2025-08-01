# [flutter] Widget - DraggableScrollableSheet: Draggable Sheet

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`DraggableScrollableSheet` adalah widget yang menampilkan lembar yang dapat diseret dan digulir dari bagian bawah layar. Pengguna dapat menyeret sheet untuk mengubah ukurannya atau menutupnya. Ini ideal untuk menampilkan konten yang mungkin lebih panjang dari layar dan memerlukan interaksi yang fleksibel.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/DraggableScrollableSheet-class.html).

## Methods and Parameters
---
`DraggableScrollableSheet` memiliki beberapa parameter penting untuk mengontrol perilakunya:
* **`builder`**: Fungsi yang mengembalikan widget yang akan ditampilkan di dalam sheet. Builder ini memberikan `BuildContext` dan `ScrollController`, yang **wajib** Anda berikan ke widget yang dapat digulir seperti `ListView`.
* **`initialChildSize`**: Ukuran awal sheet sebagai pecahan dari total ruang yang tersedia (dari 0.0 hingga 1.0).
* **`minChildSize`**: Ukuran minimal sheet saat diseret.
* **`maxChildSize`**: Ukuran maksimal sheet saat diseret.

## Best Practices or Tips
---
* **Berikan `ScrollController`**: Pastikan Anda meneruskan `ScrollController` yang disediakan oleh `builder` ke widget anak yang dapat digulir (seperti `ListView.builder` atau `SingleChildScrollView`) agar sheet dapat diseret dengan benar.
* **Atur Ukuran yang Jelas**: Tetapkan `initialChildSize`, `minChildSize`, dan `maxChildSize` dengan nilai yang logis untuk memberikan pengalaman pengguna yang intuitif.
* **Kombinasi dengan Widget Lain**: Gunakan `DraggableScrollableSheet` di dalam `Stack` untuk menempatkannya di atas konten lain, atau di dalam `showModalBottomSheet` untuk membuat sheet modal yang dapat diubah ukurannya.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const DraggableScrollableSheetExample());

class DraggableScrollableSheetExample extends StatelessWidget {
  const DraggableScrollableSheetExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('DraggableScrollableSheet Contoh')),
        body: Stack(
          children: <Widget>[
            const Center(
              child: Text('Konten Latar Belakang'),
            ),
            DraggableScrollableSheet(
              initialChildSize: 0.4,
              minChildSize: 0.2,
              maxChildSize: 0.8,
              builder: (BuildContext context, ScrollController scrollController) {
                return Container(
                  color: Colors.blue[100],
                  child: ListView.builder(
                    controller: scrollController, // Wajib
                    itemCount: 25,
                    itemBuilder: (BuildContext context, int index) {
                      return ListTile(
                        title: Text('Item $index'),
                      );
                    },
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}