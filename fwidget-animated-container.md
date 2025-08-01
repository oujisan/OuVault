# [flutter] Widget - AnimatedContainer: Animated Container

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`AnimatedContainer` adalah wadah (`Container`) yang secara otomatis menganimasikan perubahannya. Ketika Anda mengubah properti seperti warna, tinggi, atau lebar, widget ini akan melakukan transisi secara halus ke nilai baru.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/AnimatedContainer-class.html).

## Methods and Parameters
---
`AnimatedContainer` memiliki semua parameter `Container` biasa, ditambah parameter animasi:
* **`duration`**: `Duration` yang menentukan lamanya animasi berlangsung.
* **`curve`** (opsional): `Curve` yang menentukan pola animasi (misalnya, `Curves.easeIn`, `Curves.easeOut`).

Contoh properti yang dapat dianimasikan:
* **`height`, `width`**: Ukuran.
* **`color`**: Warna.
* **`alignment`**: Posisi widget anak.
* **`decoration`**: Seperti `BoxDecoration`, yang memungkinkan animasi warna, radius, dan bayangan.

## Best Practices or Tips
---
* **Widget Berstatus**: `AnimatedContainer` biasanya digunakan di dalam `StatefulWidget` karena perubahannya dipicu oleh pemanggilan `setState` saat Anda mengubah propertinya.
* **Gunakan untuk Animasi Sederhana**: Ini adalah cara termudah untuk membuat animasi implisit. Untuk animasi yang lebih kompleks atau kustom, pertimbangkan `AnimationController` dan `AnimatedBuilder`.
* **Properti yang Dapat Dianimasikan**: Hanya properti yang merupakan bagian dari `AnimatedContainer` yang akan dianimasikan. Mengubah properti di dalam `child` tidak akan memicu animasi.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const AnimatedContainerExample());

class AnimatedContainerExample extends StatefulWidget {
  const AnimatedContainerExample({Key? key}) : super(key: key);

  @override
  State<AnimatedContainerExample> createState() => _AnimatedContainerExampleState();
}

class _AnimatedContainerExampleState extends State<AnimatedContainerExample> {
  bool _isEnlarged = false;

  void _toggleSize() {
    setState(() {
      _isEnlarged = !_isEnlarged;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('AnimatedContainer Contoh')),
        body: Center(
          child: GestureDetector(
            onTap: _toggleSize,
            child: AnimatedContainer(
              duration: const Duration(seconds: 1),
              curve: Curves.easeInOut,
              height: _isEnlarged ? 200 : 100,
              width: _isEnlarged ? 200 : 100,
              decoration: BoxDecoration(
                color: _isEnlarged ? Colors.blue : Colors.red,
                borderRadius: BorderRadius.circular(_isEnlarged ? 50 : 10),
              ),
              child: const Center(
                child: Text('Ketuk Saya', style: TextStyle(color: Colors.white)),
              ),
            ),
          ),
        ),
      ),
    );
  }
}