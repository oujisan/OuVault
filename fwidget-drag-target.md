# [flutter] Widget - DragTarget: Drop Zone

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`DragTarget` adalah widget yang dapat menerima data yang diseret oleh widget `Draggable` atau `LongPressDraggable`. Widget ini mendefinisikan area di mana item dapat dijatuhkan, memungkinkan fungsionalitas drag-and-drop yang lengkap.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/DragTarget-class.html).

## Methods and Parameters
---
`DragTarget` memiliki beberapa callback yang penting untuk mengelola interaksi drop:
* **`builder`**: Wajib. Fungsi ini membangun tampilan `DragTarget` dan menerima tiga parameter: `context`, `candidateData` (data dari widget yang sedang diseret di atas target), dan `rejectedData` (data dari widget yang ditolak). Anda dapat menggunakan ini untuk memberikan umpan balik visual (misalnya, mengubah warna) saat item diseret di atasnya.
* **`onWillAccept`**: Callback yang dipanggil saat `Draggable` diseret di atas `DragTarget`. Callback ini harus mengembalikan `true` jika target akan menerima data yang diseret, dan `false` jika tidak.
* **`onAccept`**: Callback yang dipanggil saat `Draggable` dijatuhkan di atas target dan `onWillAccept` mengembalikan `true`. Ini adalah tempat Anda memproses data yang diterima.
* **`onLeave`**: Callback yang dipanggil saat `Draggable` yang sebelumnya diseret di atas target, kini keluar dari area target.

## Best Practices or Tips
---
* **Pasangkan dengan `Draggable`**: `DragTarget` tidak akan berfungsi tanpa widget `Draggable` atau `LongPressDraggable` yang menyediakan data untuk diseret.
* **Berikan Umpan Balik Visual**: Gunakan parameter `builder` untuk mengubah tampilan `DragTarget` saat item diseret di atasnya. Ini akan memberikan pengalaman pengguna yang lebih baik.
* **Validasi Data**: Gunakan `onWillAccept` untuk memvalidasi jenis data yang diseret. Misalnya, jika target hanya menerima string, Anda bisa mengembalikan `false` jika data yang diseret bukan string.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const DragTargetExample());

class DragTargetExample extends StatefulWidget {
  const DragTargetExample({Key? key}) : super(key: key);

  @override
  State<DragTargetExample> createState() => _DragTargetExampleState();
}

class _DragTargetExampleState extends State<DragTargetExample> {
  Color _dragTargetColor = Colors.grey[200]!;
  int _droppedData = 0;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('DragTarget Contoh')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Draggable<int>(
                data: 10,
                feedback: Container(
                  height: 100,
                  width: 100,
                  color: Colors.blue.withOpacity(0.5),
                  child: const Center(child: Text('10', style: TextStyle(color: Colors.white, fontSize: 24))),
                ),
                childWhenDragging: Container(
                  height: 100,
                  width: 100,
                  color: Colors.grey,
                  child: const Center(child: Text('10')),
                ),
                child: Container(
                  height: 100,
                  width: 100,
                  color: Colors.blue,
                  child: const Center(child: Text('Seret')),
                ),
              ),
              const SizedBox(height: 50),
              DragTarget<int>(
                onWillAccept: (data) {
                  setState(() => _dragTargetColor = Colors.green[200]!);
                  return true;
                },
                onAccept: (data) {
                  setState(() {
                    _droppedData = data;
                    _dragTargetColor = Colors.grey[200]!;
                  });
                },
                onLeave: (data) {
                  setState(() => _dragTargetColor = Colors.grey[200]!);
                },
                builder: (BuildContext context, List<int?> candidateData, List rejectedData) {
                  return Container(
                    height: 150,
                    width: 150,
                    color: _dragTargetColor,
                    child: Center(
                      child: Text('Drop Here\n(Dropped: $_droppedData)', textAlign: TextAlign.center),
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}