# [flutter] Widget - CircleAvatar: Circular Avatar

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`CircleAvatar` adalah widget yang menampilkan avatar melingkar. Ini sering digunakan untuk gambar profil pengguna, inisial nama, atau placeholder.

Lihat dokumentasi resmi Flutter di [https://api.flutter.dev/flutter/material/CircleAvatar-class.html](https://api.flutter.dev/flutter/material/CircleAvatar-class.html).

## Methods and Parameters
---
`CircleAvatar` memiliki beberapa parameter untuk menentukan konten dan tampilannya:
* **`child`** (opsional): Widget yang ditampilkan di tengah lingkaran, biasanya `Text` (untuk inisial) atau `Icon`.
* **`backgroundImage`** (opsional): Menggunakan `ImageProvider` untuk menampilkan gambar di latar belakang. Ini adalah cara yang umum untuk menampilkan gambar profil.
* **`backgroundColor`** (opsional): Mengatur warna latar belakang lingkaran jika tidak ada gambar.
* **`radius`** (opsional): Mengontrol ukuran lingkaran.

## Best Practices or Tips
---
* **Penggunaan yang Fleksibel**: Anda dapat menggunakan `CircleAvatar` dengan gambar (menggunakan `backgroundImage`), teks (menggunakan `child: Text`), atau ikon (menggunakan `child: Icon`) untuk berbagai kasus penggunaan.
* **Penyedia Gambar**: Gunakan `NetworkImage` untuk memuat gambar dari internet atau `AssetImage` untuk gambar lokal.
* **Placeholder**: Saat memuat gambar dari jaringan, Anda dapat menempatkan `Text` atau `Icon` di `child` untuk berfungsi sebagai placeholder hingga gambar dimuat.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const CircleAvatarExample());

class CircleAvatarExample extends StatelessWidget {
  const CircleAvatarExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('CircleAvatar Contoh')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              CircleAvatar(
                radius: 50,
                backgroundImage: NetworkImage('[https://picsum.photos/200](https://picsum.photos/200)'),
              ),
              SizedBox(height: 20),
              CircleAvatar(
                radius: 40,
                backgroundColor: Colors.blue,
                child: Text('A', style: TextStyle(color: Colors.white, fontSize: 30)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}