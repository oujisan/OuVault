# [flutter] Widget - AnimatedDefaultTextStyle: Animated Text Style

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`AnimatedDefaultTextStyle` adalah widget yang secara otomatis menganimasikan transisi dari gaya teks lama ke gaya teks baru. Ini berguna untuk memberikan efek visual yang halus saat Anda mengubah properti teks seperti warna, ukuran font, atau berat.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/AnimatedDefaultTextStyle-class.html).

## Methods and Parameters
---
`AnimatedDefaultTextStyle` memiliki beberapa parameter untuk mengontrol animasi dan gaya teksnya:
* **`child`**: Widget yang berisi teks yang akan dianimasikan (biasanya widget `Text`).
* **`style`**: `TextStyle` yang akan menjadi gaya target animasi.
* **`duration`**: `Duration` yang menentukan lamanya animasi berlangsung.
* **`curve`** (opsional): `Curve` yang menentukan pola animasi.

## Best Practices or Tips
---
* **Animasi Teks Sederhana**: Ini adalah cara yang paling sederhana untuk menganimasikan gaya teks. Cukup ubah properti `style` di dalam `setState` dan `AnimatedDefaultTextStyle` akan mengurus transisinya.
* **Gunakan untuk Perubahan State**: Ideal untuk situasi di mana gaya teks berubah sebagai respons terhadap interaksi pengguna, seperti saat tombol ditekan atau nilai berubah.
* **Hindari Penggunaan Berlebihan**: Gunakan animasi dengan bijak. Animasi yang terlalu sering atau tidak perlu dapat mengganggu pengalaman pengguna.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const AnimatedDefaultTextStyleExample());

class AnimatedDefaultTextStyleExample extends StatefulWidget {
  const AnimatedDefaultTextStyleExample({Key? key}) : super(key: key);

  @override
  State<AnimatedDefaultTextStyleExample> createState() => _AnimatedDefaultTextStyleExampleState();
}

class _AnimatedDefaultTextStyleExampleState extends State<AnimatedDefaultTextStyleExample> {
  bool _isLarge = false;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('AnimatedDefaultTextStyle Contoh')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              AnimatedDefaultTextStyle(
                duration: const Duration(milliseconds: 500),
                curve: Curves.easeInOut,
                style: TextStyle(
                  fontSize: _isLarge ? 32 : 16,
                  fontWeight: _isLarge ? FontWeight.bold : FontWeight.normal,
                  color: _isLarge ? Colors.blue : Colors.red,
                ),
                child: const Text('Teks yang Berubah'),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _isLarge = !_isLarge;
                  });
                },
                child: const Text('Ganti Gaya'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}