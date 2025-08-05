# [flutter] Widget - DrawerHeader: Drawer Header

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`DrawerHeader` adalah widget yang dirancang khusus untuk ditempatkan di bagian atas `Drawer`. Ini menyediakan ruang header yang rapi, sering digunakan untuk menampilkan informasi profil pengguna, nama aplikasi, atau gambar latar belakang.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/material/DrawerHeader-class.html).

## Methods and Parameters
---
* **`child`**: Widget yang akan ditempatkan di dalam header, biasanya `Text`, `CircleAvatar`, atau `Column` untuk menggabungkan beberapa widget.
* **`decoration`** (opsional): Menggunakan `BoxDecoration` untuk memberikan latar belakang yang kustom, seperti warna solid atau gambar.

## Best Practices or Tips
---
* **Gunakan Sebagai Item Pertama**: Selalu tempatkan `DrawerHeader` sebagai item pertama di dalam `ListView` dari `Drawer` Anda.
* **Atur Margin `ListView`**: Setel `padding: EdgeInsets.zero` pada `ListView` yang membungkus konten `Drawer` untuk memastikan `DrawerHeader` mengisi seluruh lebar dan tidak memiliki margin atas.
* **Kustomisasi Konten**: Anda dapat menempatkan widget apa pun di dalam `child`, seperti `Row` dengan gambar profil dan nama pengguna, untuk membuat header yang lebih fungsional.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const DrawerHeaderExample());

class DrawerHeaderExample extends StatelessWidget {
  const DrawerHeaderExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('DrawerHeader Contoh')),
        drawer: Drawer(
          child: ListView(
            padding: EdgeInsets.zero, // Penting untuk menghilangkan padding
            children: <Widget>[
              const DrawerHeader(
                decoration: BoxDecoration(
                  color: Colors.blue,
                ),
                child: Text(
                  'Halo, Pengguna!',
                  style: TextStyle(color: Colors.white, fontSize: 24),
                ),
              ),
              ListTile(
                leading: const Icon(Icons.person),
                title: const Text('Profil'),
                onTap: () {
                  Navigator.pop(context);
                },
              ),
            ],
          ),
        ),
        body: const Center(
          child: Text('Abaikan'),
        ),
      ),
    );
  }
}