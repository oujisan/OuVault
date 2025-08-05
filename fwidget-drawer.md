# [flutter] Widget - Drawer: Navigation Panel

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Drawer` adalah panel navigasi material yang meluncur dari sisi layar (biasanya dari kiri). Ini digunakan untuk menampilkan menu navigasi utama aplikasi, seperti daftar halaman, profil pengguna, atau pengaturan.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/material/Drawer-class.html).

## Methods and Parameters
---
`Drawer` adalah properti dari `Scaffold`. Anda mengaturnya dengan parameter `drawer`.
* **`child`**: Widget yang berada di dalam `Drawer`, yang biasanya berupa `ListView`.
* **`elevation`** (opsional): Mengatur ketinggian `Drawer` (bayangan).
* **`width`** (opsional): Mengatur lebar `Drawer`.

Untuk membuka `Drawer`, pengguna dapat menggeser dari sisi layar atau menekan ikon `menu` di `AppBar`. Untuk menutupnya, pengguna dapat menggeser ke arah sebaliknya atau mengetuk di luar `Drawer`.

## Best Practices or Tips
---
* **Gunakan `ListView` di Dalamnya**: Selalu bungkus konten `Drawer` Anda dengan `ListView` untuk membuatnya dapat digulir jika kontennya melebihi tinggi layar.
* **Tambahkan `DrawerHeader`**: Gunakan `DrawerHeader` sebagai item pertama di `ListView` untuk memberikan area header yang rapi, sering kali untuk menampilkan nama pengguna atau logo.
* **Gunakan `ListTile` untuk Menu**: Gunakan `ListTile` untuk setiap item menu. Ini memberikan antarmuka yang bersih dan mudah ditekan.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const DrawerExample());

class DrawerExample extends StatelessWidget {
  const DrawerExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Drawer Contoh'),
        ),
        drawer: Drawer(
          child: ListView(
            padding: EdgeInsets.zero,
            children: const <Widget>[
              DrawerHeader(
                decoration: BoxDecoration(
                  color: Colors.blue,
                ),
                child: Text('Menu Navigasi', style: TextStyle(color: Colors.white, fontSize: 24)),
              ),
              ListTile(
                leading: Icon(Icons.home),
                title: Text('Beranda'),
              ),
              ListTile(
                leading: Icon(Icons.settings),
                title: Text('Pengaturan'),
              ),
            ],
          ),
        ),
        body: const Center(
          child: Text('Tekan ikon menu di AppBar untuk membuka Drawer'),
        ),
      ),
    );
  }
}