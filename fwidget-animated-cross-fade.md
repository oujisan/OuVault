# [flutter] Widget - AnimatedCrossFade: Cross-fade Animation

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`AnimatedCrossFade` adalah widget yang menganimasikan transisi antara dua widget anak (`firstChild` dan `secondChild`) dengan efek cross-fade yang halus. Ini sangat berguna ketika Anda ingin beralih antara dua tampilan yang berbeda dengan transisi yang elegan.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/AnimatedCrossFade-class.html).

## Methods and Parameters
---
`AnimatedCrossFade` memiliki beberapa parameter penting:
* **`firstChild`**: Widget yang ditampilkan saat transisi dalam keadaan `CrossFadeState.showFirst`.
* **`secondChild`**: Widget yang ditampilkan saat transisi dalam keadaan `CrossFadeState.showSecond`.
* **`crossFadeState`**: `CrossFadeState` yang mengontrol widget mana yang saat ini terlihat. Ini bisa berupa `CrossFadeState.showFirst` atau `CrossFadeState.showSecond`.
* **`duration`**: `Duration` yang menentukan lamanya animasi berlangsung.
* **`firstCurve`** dan **`secondCurve`** (opsional): Mengatur kurva animasi untuk setiap widget.

## Best Practices or Tips
---
* **Gunakan untuk Dua Keadaan**: `AnimatedCrossFade` dirancang khusus untuk transisi antara dua keadaan. Jika Anda memiliki lebih dari dua keadaan, pertimbangkan `AnimatedSwitcher` atau `AnimatedContainer`.
* **Kontrol Transisi dengan `setState`**: Perubahan `crossFadeState` harus dipicu oleh pemanggilan `setState` di dalam `StatefulWidget`.
* **Ukuran yang Sama**: Pastikan kedua widget anak (`firstChild` dan `secondChild`) memiliki ukuran yang serupa atau setidaknya terkelola dengan baik untuk menghindari lonjakan tata letak yang tiba-tiba selama animasi.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const AnimatedCrossFadeExample());

class AnimatedCrossFadeExample extends StatefulWidget {
  const AnimatedCrossFadeExample({Key? key}) : super(key: key);

  @override
  State<AnimatedCrossFadeExample> createState() => _AnimatedCrossFadeExampleState();
}

class _AnimatedCrossFadeExampleState extends State<AnimatedCrossFadeExample> {
  bool _showFirst = true;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('AnimatedCrossFade Contoh')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              AnimatedCrossFade(
                duration: const Duration(seconds: 1),
                crossFadeState: _showFirst
                    ? CrossFadeState.showFirst
                    : CrossFadeState.showSecond,
                firstChild: Container(
                  width: 200,
                  height: 200,
                  color: Colors.blue,
                  child: const Center(child: Text('Widget Pertama')),
                ),
                secondChild: Container(
                  width: 200,
                  height: 200,
                  color: Colors.red,
                  child: const Center(child: Text('Widget Kedua')),
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _showFirst = !_showFirst;
                  });
                },
                child: const Text('Ganti Widget'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}