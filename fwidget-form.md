# [flutter] Widget - Form: Form Container

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Form` adalah widget yang berfungsi sebagai wadah untuk mengelompokkan beberapa widget input (seperti `TextFormField`). `Form` memungkinkan Anda untuk memvalidasi dan menyimpan semua bidang input di dalamnya secara serentak.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/Form-class.html).

## Methods and Parameters
---
* **`child`**: Widget yang berisi bidang-bidang formulir Anda.
* **`key`**: `GlobalKey<FormState>`. Anda harus membuat kunci unik ini untuk dapat mengakses status `Form` dari luar.
* **`onChanged`** (opsional): Callback yang dipanggil saat bidang input di dalam formulir berubah.

Metode yang paling penting untuk digunakan adalah:
* **`_formKey.currentState?.validate()`**: Memvalidasi semua bidang di dalam formulir.
* **`_formKey.currentState?.save()`**: Menyimpan semua bidang yang valid.

## Best Practices or Tips
---
* **Gunakan dengan `GlobalKey`**: Selalu berikan `GlobalKey<FormState>` ke `Form` untuk dapat mengakses statusnya (validasi, simpan) dari tombol atau widget lain.
* **Pasangkan dengan `TextFormField`**: `Form` dirancang untuk bekerja dengan widget `FormField`, seperti `TextFormField`, yang memiliki properti validasi dan penyimpanan bawaan.
* **Peringatan**: Metode validasi dan penyimpanan hanya akan berfungsi jika Anda memanggilnya setelah formulir dibuat dan kuncinya telah ditetapkan.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const FormExample());

class FormExample extends StatefulWidget {
  const FormExample({Key? key}) : super(key: key);

  @override
  State<FormExample> createState() => _FormExampleState();
}

class _FormExampleState extends State<FormExample> {
  final _formKey = GlobalKey<FormState>();
  String _name = '';

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Form Contoh')),
        body: Form(
          key: _formKey,
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: <Widget>[
                TextFormField(
                  decoration: const InputDecoration(labelText: 'Nama'),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Nama tidak boleh kosong';
                    }
                    return null;
                  },
                  onSaved: (value) {
                    _name = value!;
                  },
                ),
                ElevatedButton(
                  onPressed: () {
                    if (_formKey.currentState!.validate()) {
                      _formKey.currentState!.save();
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('Form berhasil disubmit, nama: $_name')),
                      );
                    }
                  },
                  child: const Text('Submit'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}