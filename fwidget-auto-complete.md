# [flutter] Widget - Autocomplete: Autocomplete Input

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Autocomplete` adalah widget yang menyediakan saran pelengkapan otomatis saat pengguna mengetik. Widget ini sangat berguna untuk kolom input seperti pencarian atau formulir dengan daftar opsi yang sudah ditentukan.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/Autocomplete-class.html).

## Methods and Parameters
---
`Autocomplete` memiliki beberapa parameter kunci untuk mengontrol perilakunya:
* **`optionsBuilder`**: Wajib. Callback yang dipanggil setiap kali teks input berubah. Callback ini harus mengembalikan `Iterable<T>` dari opsi yang cocok dengan input.
* **`onSelected`**: Callback yang dipanggil saat pengguna memilih salah satu opsi dari daftar.
* **`fieldViewBuilder`**: Wajib. Fungsi ini membangun widget input teks yang digunakan oleh `Autocomplete`. Anda dapat menggunakannya untuk mengustomisasi tampilan `TextFormField` Anda.
* **`optionsViewBuilder`** (opsional): Fungsi ini membangun tampilan daftar saran. Anda dapat mengustomisasi bagaimana daftar saran ditampilkan.

## Best Practices or Tips
---
* **Buat Opsi Secara Dinamis**: Gunakan `optionsBuilder` untuk memfilter daftar opsi yang besar berdasarkan input pengguna. Ini akan membuat aplikasi Anda lebih responsif.
* **Kustomisasi UI**: Gunakan `fieldViewBuilder` dan `optionsViewBuilder` untuk memastikan tampilan `Autocomplete` sesuai dengan desain aplikasi Anda.
* **Penggunaan yang Efisien**: Jika daftar opsi sangat besar, pertimbangkan untuk menunda pemfilteran hingga pengguna berhenti mengetik sebentar (debouncing) untuk menghemat sumber daya.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const AutocompleteExample());

class AutocompleteExample extends StatelessWidget {
  const AutocompleteExample({Key? key}) : super(key: key);

  static const List<String> _options = <String>[
    'apple',
    'banana',
    'grape',
    'orange',
    'pineapple',
  ];

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Autocomplete Contoh')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Autocomplete<String>(
              optionsBuilder: (TextEditingValue textEditingValue) {
                if (textEditingValue.text == '') {
                  return const Iterable<String>.empty();
                }
                return _options.where((String option) {
                  return option.contains(textEditingValue.text.toLowerCase());
                });
              },
              onSelected: (String selection) {
                print('Anda memilih $selection');
              },
            ),
          ),
        ),
      ),
    );
  }
}