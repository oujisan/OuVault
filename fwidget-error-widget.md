# [flutter] Widget - ErrorWidget: Error Display

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`ErrorWidget` adalah widget yang ditampilkan oleh Flutter secara internal ketika terjadi kesalahan fatal yang tidak tertangani di pohon widget. Anda tidak perlu menggunakan widget ini secara langsung. Namun, memahami fungsinya penting untuk penanganan kesalahan yang baik.

Anda dapat mengganti `ErrorWidget` default dengan versi kustom menggunakan `ErrorWidget.builder`.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/ErrorWidget-class.html).

## Methods and Parameters
---
Anda tidak membuat instance `ErrorWidget` secara langsung. Sebaliknya, Anda dapat mengontrol tampilannya secara global dengan mengatur properti `ErrorWidget.builder`.
* **`ErrorWidget.builder`**: Ini adalah properti statis yang menerima callback yang mengembalikan widget `ErrorWidget` kustom Anda. Callback ini akan dipanggil saat terjadi kesalahan.

## Best Practices or Tips
---
* **Penanganan Kesalahan Global**: Gunakan `ErrorWidget.builder` di awal aplikasi Anda (misalnya di fungsi `main`) untuk mengatur tampilan kesalahan global. Ini akan membantu Anda memberikan pengalaman pengguna yang lebih baik daripada hanya menampilkan layar merah yang penuh dengan teks error.
* **Sertakan Informasi Bantuan**: Widget kesalahan kustom Anda harus memberikan pesan yang lebih ramah pengguna dan mungkin informasi kontak atau tombol "laporkan bug".
* **Hindari `ErrorWidget` Langsung**: Anda tidak perlu menggunakan `ErrorWidget` sebagai bagian dari tata letak Anda. Jika Anda ingin menampilkan pesan kesalahan secara manual, gunakan widget normal seperti `Text` atau `Icon`.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() {
  ErrorWidget.builder = (FlutterErrorDetails details) {
    // Logika kustom untuk penanganan kesalahan
    print('Terjadi kesalahan tidak tertangani: ${details.exception}');
    return const Center(
      child: Text(
        'Oops! Terjadi kesalahan. Silakan coba lagi.',
        style: TextStyle(color: Colors.red, fontSize: 18),
        textAlign: TextAlign.center,
      ),
    );
  };
  runApp(const ErrorWidgetExample());
}

class ErrorWidgetExample extends StatelessWidget {
  const ErrorWidgetExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('ErrorWidget Contoh')),
        body: Center(
          child: ElevatedButton(
            onPressed: () {
              // Kode ini akan sengaja menimbulkan error
              throw Exception('Ini adalah error sengaja!');
            },
            child: const Text('Timbulkan Error'),
          ),
        ),
      ),
    );
  }
}