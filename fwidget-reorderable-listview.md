# [flutter] Widget - ReorderableListView: Reorderable List

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`ReorderableListView` membuat daftar item yang dapat digulir dan item-item di dalamnya dapat diatur ulang (reorder) oleh pengguna melalui interaksi menyeret.

Lihat dokumentasi resmi Flutter di [https://api.flutter.dev/flutter/material/ReorderableListView-class.html](https://api.flutter.dev/flutter/material/ReorderableListView-class.html).

## Methods and Parameters
---
* **`onReorder`**: Sebuah callback yang dipanggil ketika pengguna selesai mengatur ulang item. Callback ini menerima dua parameter: `oldIndex` (indeks item sebelum diseret) dan `newIndex` (indeks item setelah dijatuhkan). Anda harus memperbarui model data Anda di dalam callback ini.
* **`children`**: Daftar widget anak yang dapat diatur ulang. **Setiap anak harus memiliki `Key` unik** agar Flutter dapat mengidentifikasi setiap item dengan benar.
* **`buildDefaultDragHandles`** (opsional): Jika `true` (default), widget akan secara otomatis menyediakan area pegangan seret di sisi kiri setiap item.

## Best Practices or Tips
---
* **Wajib Menggunakan `Key`**: Pastikan setiap item di dalam daftar `children` memiliki `Key` yang unik, seperti `ValueKey`, untuk memastikan Flutter dapat melacak item dengan benar selama proses pengurutan ulang.
* **Perbarui Model Data**: Logika pengurutan ulang item tidak terjadi secara otomatis. Anda harus mengimplementasikan logika untuk memperbarui daftar data Anda di dalam callback `onReorder`.
* **Gunakan Widget ListTile**: `ReorderableListView` sering digunakan dengan `ListTile` karena `ListTile` menyediakan struktur yang sudah jadi untuk daftar item.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const ReorderableListViewExample());

class ReorderableListViewExample extends StatefulWidget {
  const ReorderableListViewExample({Key? key}) : super(key: key);

  @override
  State<ReorderableListViewExample> createState() => _ReorderableListViewExampleState();
}

class _ReorderableListViewExampleState extends State<ReorderableListViewExample> {
  final List<int> _items = List<int>.generate(10, (i) => i);

  void _onReorder(int oldIndex, int newIndex) {
    setState(() {
      if (oldIndex < newIndex) {
        newIndex -= 1;
      }
      final int item = _items.removeAt(oldIndex);
      _items.insert(newIndex, item);
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('ReorderableListView Contoh')),
        body: ReorderableListView(
          onReorder: _onReorder,
          children: <Widget>[
            for (int index = 0; index < _items.length; index += 1)
              ListTile(
                key: Key('$index'), // Kunci unik wajib
                tileColor: _items[index].isOdd ? Colors.blue.shade100 : Colors.blue.shade300,
                title: Text('Item ${_items[index]}'),
                trailing: const Icon(Icons.drag_handle),
              ),
          ],
        ),
      ),
    );
  }
}