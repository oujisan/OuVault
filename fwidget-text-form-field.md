# [flutter] Widget - TextFormField: Text Input

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`TextFormField` adalah widget bidang input teks yang merupakan turunan dari `TextField` tetapi dirancang khusus untuk digunakan di dalam `Form`. Widget ini memiliki fitur validasi dan penyimpanan data yang terintegrasi dengan `Form`.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/material/TextFormField-class.html).

## Methods and Parameters
---
`TextFormField` memiliki banyak parameter dari `TextField` dan beberapa tambahan untuk integrasi dengan `Form`:
* **`decoration`**: Menggunakan `InputDecoration` untuk menambahkan label, placeholder, ikon, dan gaya visual lainnya.
* **`validator`**: Callback yang dipanggil saat `Form` divalidasi. Callback ini menerima nilai input dan harus mengembalikan string pesan error jika validasi gagal, atau `null` jika valid.
* **`onSaved`**: Callback yang dipanggil saat `Form` dipanggil `save()`. Ini adalah tempat untuk menyimpan nilai input ke dalam variabel.
* **`controller`** (opsional): Menggunakan `TextEditingController` untuk mengontrol teks secara programatik.

## Best Practices or Tips
---
* **Gunakan di dalam `Form`**: Selalu gunakan `TextFormField` di dalam `Form` untuk memanfaatkan fitur validasi dan penyimpanan secara maksimal.
* **Berikan Pesan Validasi yang Jelas**: Pesan yang dikembalikan dari `validator` harus informatif dan mudah dipahami oleh pengguna.
* **Integrasi `onSaved`**: Gunakan `onSaved` untuk menyimpan nilai setelah validasi berhasil. Hindari menggunakan `onChanged` untuk tujuan ini karena `onChanged` dipanggil setiap kali teks berubah, yang bisa tidak efisien.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const TextFormFieldExample());

class TextFormFieldExample extends StatefulWidget {
  const TextFormFieldExample({Key? key}) : super(key: key);

  @override
  State<TextFormFieldExample> createState() => _TextFormFieldExampleState();
}

class _TextFormFieldExampleState extends State<TextFormFieldExample> {
  final _formKey = GlobalKey<FormState>();
  String _email = '';

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('TextFormField Contoh')),
        body: Form(
          key: _formKey,
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: <Widget>[
                TextFormField(
                  decoration: const InputDecoration(labelText: 'Email'),
                  validator: (value) {
                    if (value == null || value.isEmpty || !value.contains('@')) {
                      return 'Masukkan email yang valid';
                    }
                    return null;
                  },
                  onSaved: (value) {
                    _email = value!;
                  },
                ),
                ElevatedButton(
                  onPressed: () {
                    if (_formKey.currentState!.validate()) {
                      _formKey.currentState!.save();
                      print('Email yang disimpan: $_email');
                    }
                  },
                  child: const Text('Simpan'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}