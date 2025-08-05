# [flutter] Widget - Draggable: Drag Source

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Draggable` adalah widget yang dapat diseret oleh pengguna. Ini adalah komponen kunci dari sistem drag-and-drop di Flutter, yang menyediakan data untuk `DragTarget`. Widget ini secara visual menempel pada jari pengguna saat diseret.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/Draggable-class.html).

## Methods and Parameters
---
`Draggable` memiliki beberapa parameter penting untuk mengontrol perilakunya:
* **`child`**: Widget yang ditampilkan saat tidak diseret.
* **`feedback`**: Widget yang ditampilkan selama penyeretan. Biasanya, ini adalah versi visual yang berbeda atau lebih ringan dari `child`.
* **`data`**: Objek data yang diteruskan ke `DragTarget` saat dijatuhkan.
* **`childWhenDragging`** (opsional): Widget yang ditampilkan di posisi `child` asli saat penyeretan sedang berlangsung. Ini bisa berupa placeholder kosong atau versi yang lebih redup.
* **`onDragStarted`**, **`onDragEnd`**, **`onDragCompleted`**: Callback untuk mendeteksi berbagai fase penyeretan.

## Best Practices or Tips
---
* **Berikan Umpan Balik yang Jelas**: Widget `feedback` harus memberikan petunjuk visual yang jelas bahwa item sedang diseret. Menggunakan transparansi atau bayangan adalah praktik yang baik.
* **Gunakan `childWhenDragging`**: Mengatur `childWhenDragging` ke placeholder kosong atau versi yang lebih redup dapat memberikan efek visual yang lebih baik, menunjukkan kepada pengguna bahwa item asli telah dipindahkan.
* **Pasangkan dengan `DragTarget`**: Agar drag-and-drop berfungsi penuh, selalu gunakan `Draggable` bersama dengan `DragTarget`.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const DraggableExample());

class DraggableExample extends StatelessWidget {
  const DraggableExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Draggable Contoh')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Draggable<int>(
                data: 5,
                childWhenDragging: Container(
                  height: 100,
                  width: 100,
                  color: Colors.grey,
                  child: const Center(child: Text('Placeholder')),
                ),
                feedback: Container(
                  height: 100,
                  width: 100,
                  color: Colors.blue.withOpacity(0.5),
                  child: const Center(child: Text('5', style: TextStyle(color: Colors.white))),
                ),
                child: Container(
                  height: 100,
                  width: 100,
                  color: Colors.blue,
                  child: const Center(child: Text('Seret')),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}