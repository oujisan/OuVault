# [flutter] Tips - Device Preview Hidden Devices (Samsung S25)

![device_preview](https://res.cloudinary.com/dz1h6jiye/image/upload/v1753705855/device_preview_r6z5dv.png)

[Device Preview](https://pub.dev/packages/device_preview) adalah library dari Flutter untuk menampilkan debug aplikasi di berbagai tampilan perangkat mulai dari Handphone, Tablet, dan PC. Dalam memilih device terkadang kita tidak menemukan device yang kita inginkan contohnya Samsung Galaxy S25.
Ternyata menu device pada device preview belum menampilkan seluruh device tersedia yang dapat kita coba. sebagai contoh, series samsung yang ada hanya sampai Samsung Galay S20.

Kamu bisa menemukan pilihan Samsung Galaxy S25 pada IntelliSense code.
![screenshoot_code](https://res.cloudinary.com/dz1h6jiye/image/upload/v1753707861/Screenshot_2025-07-28_195049_ekqpfr.png)

Apabila belum mempunyai IntelliSense, install ekstensi pada menu `Extensions` Visual Studio Code yang bernama [Flutter Widget Snippets](https://marketplace.visualstudio.com/items?itemName=alexisvt.flutter-snippets).

Beikut code pada `main.dart`:
```dart
// main.dart
import 'package:flutter/material.dart';
import 'package:device_preview/device_preview.dart';
import 'package:flutter/foundation.dart';

void main() {
  runApp(
    DevicePreview(
      enabled: !kReleaseMode,
      builder: (context) => const MainApp(),
      defaultDevice: Devices.android.samsungGalaxyS25,
    ),
  );
}
. . .
```

Menggunakan parameter `defaultDevices`, kita menetapkan devices yang akan muncul ketika debug adalah Samsung Galaxy S25. Kita bisa eksplorasi dengan mengganti properti `android` dengan `ios`, `linux`, `macOS` atau `windows`.

Dengan ini device yang akan muncul ketika `debug` adalah Samsung Galaxy S25, tentu saja kita masih bisa mengganti device ketika debug seperti biasa.

![debug](https://res.cloudinary.com/dz1h6jiye/image/upload/v1753707897/Screenshot_2025-07-28_194828_p0y56z.png)
