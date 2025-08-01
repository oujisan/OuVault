# [flutter] Widget - Dismissible: Swipe to Dismiss

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Dismissible` adalah widget yang memungkinkan pengguna untuk menutup (dismiss) atau menghapus widget anaknya dengan menggesernya ke samping. Ini sangat umum digunakan untuk menghapus item dari daftar (list).

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/Dismissible-class.html).

## Methods and Parameters
---
`Dismissible` memiliki parameter utama berikut:
* **`key`**: **Parameter wajib** dan harus unik untuk setiap `Dismissible`. Flutter menggunakan `key` ini untuk melacak dan mengidentifikasi widget.
* **`child`**: Widget yang dapat digeser untuk ditutup.
* **`onDismissed`**: Callback yang dipanggil saat widget telah digeser sepenuhnya. Callback ini menerima `DismissDirection` yang menunjukkan arah geser (`left`, `right`, dll.).
* **`background`** (opsional): Widget yang ditampilkan di belakang item saat digeser. Ini sering digunakan untuk menunjukkan aksi yang akan dilakukan, seperti ikon hapus.

## Best Practices or Tips
---
* **Gunakan `Key` Unik**: Menggunakan `Key` yang unik (misalnya `ValueKey`) sangat penting agar `Dismissible` berfungsi dengan benar di dalam daftar yang dinamis.
* **Perbarui Model Data**: Di dalam callback `onDismissed`, Anda harus memanggil `setState` untuk menghapus item dari daftar model data Anda. Jika tidak, item akan tetap ada dalam data meskipun sudah tidak terlihat di UI.
* **Tambahkan Konfirmasi**: Untuk menghindari penghapusan yang tidak disengaja, Anda dapat menggunakan parameter `confirmDismiss` yang dapat mengembalikan `Future<bool>` untuk menampilkan dialog konfirmasi sebelum item benar-benar dihapus.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const DismissibleExample());

class DismissibleExample extends StatefulWidget {
  const DismissibleExample({Key? key}) : super(key: key);

  @override
  State<DismissibleExample> createState() => _DismissibleExampleState();
}

class _DismissibleExampleState extends State<DismissibleExample> {
  final List<String> items = List<String>.generate(10, (i) => 'Item $i');

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Dismissible Contoh')),
        body: ListView.builder(
          itemCount: items.length,
          itemBuilder: (context, index) {
            final item = items[index];
            return Dismissible(
              key: Key(item), // Kunci unik wajib
              onDismissed: (direction) {
                setState(() {
                  items.removeAt(index);
                });
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('$item ditutup')),
                );
              },
              background: Container(
                color: Colors.red,
                alignment: Alignment.centerRight,
                padding: const EdgeInsets.symmetric(horizontal: 20.0),
                child: const Icon(Icons.delete, color: Colors.white),
              ),
              child: ListTile(
                title: Text(item),
              ),
            );
          },
        ),
      ),
    );
  }
}