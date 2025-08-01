# [flutter] Widget - LongPressDraggable: Drag and Drop

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
Widget ini memungkinkan widget anak untuk diseret (drag) setelah ditekan dan ditahan (long press). Ini adalah bagian dari sistem drag-and-drop di Flutter, yang biasanya digunakan bersama dengan `DragTarget`.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/LongPressDraggable-class.html).

## Methods and Parameters
---
`LongPressDraggable` memiliki beberapa parameter penting untuk mengontrol perilakunya:
* **`child`**: Widget yang akan diseret.
* **`feedback`**: Widget yang ditampilkan selama proses penyeretan. Biasanya, ini adalah salinan visual dari `child`.
* **`data`**: Objek data yang diteruskan ke `DragTarget` saat widget dijatuhkan.
* **`onDragStarted`**: Callback yang dipanggil saat penyeretan dimulai.
* **`onDragEnd`**: Callback yang dipanggil saat penyeretan berakhir.

## Best Practices or Tips
---
* **Gunakan `DragTarget`**: Untuk fungsionalitas drag-and-drop yang lengkap, pasangkan `LongPressDraggable` dengan `DragTarget` yang akan menerima data yang diseret.
* **Feedback yang Jelas**: Pastikan widget `feedback` memiliki visual yang jelas dan berbeda dari `child` aslinya, seperti transparansi atau bayangan, untuk memberikan umpan balik visual kepada pengguna.
* **Data yang Tepat**: Gunakan parameter `data` untuk meneruskan informasi penting, seperti ID atau objek, yang dibutuhkan oleh `DragTarget` untuk memproses item yang dijatuhkan.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const LongPressDraggableExample());

class LongPressDraggableExample extends StatelessWidget {
  const LongPressDraggableExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('LongPressDraggable Contoh')),
        body: Center(
          child: DragTarget<int>(
            onWillAccept: (data) => true,
            onAccept: (data) {
              // Logika saat item diterima
              print('Dropped data: $data');
            },
            builder: (context, candidateData, rejectedData) {
              return LongPressDraggable<int>(
                data: 1, // Data yang akan diserahkan
                feedback: Container(
                  height: 100,
                  width: 100,
                  color: Colors.blue.withOpacity(0.5),
                  child: const Icon(Icons.star, size: 50, color: Colors.white),
                ),
                child: Container(
                  height: 100,
                  width: 100,
                  color: Colors.blue,
                  child: const Icon(Icons.star, size: 50, color: Colors.white),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}