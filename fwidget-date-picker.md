# [flutter] Widget - DatePicker: Date Picker

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`showDatePicker` adalah fungsi yang menampilkan dialog pop-up yang memungkinkan pengguna untuk memilih satu tanggal. Ini ideal untuk formulir atau fitur yang memerlukan input tanggal yang spesifik, seperti tanggal lahir atau tanggal reservasi.

Lihat dokumentasi resmi Flutter di [https://api.flutter.dev/flutter/material/showDatePicker.html](https://api.flutter.dev/flutter/material/showDatePicker.html).

## Methods and Parameters
---
Untuk menampilkan dialog `DatePicker`, Anda menggunakan fungsi **`showDatePicker`**. Parameter yang paling penting adalah:
* **`context`**: Konteks build dari widget saat ini.
* **`initialDate`**: Tanggal yang akan ditampilkan saat dialog pertama kali dibuka.
* **`firstDate`**: Tanggal paling awal yang dapat dipilih.
* **`lastDate`**: Tanggal paling akhir yang dapat dipilih.

Fungsi `showDatePicker` mengembalikan `Future<DateTime?>`, yang berarti Anda dapat menggunakan `await` untuk mendapatkan tanggal yang dipilih oleh pengguna.

## Best Practices or Tips
---
* **Atur Rentang Tanggal**: Selalu tentukan `firstDate` dan `lastDate` untuk membatasi pilihan pengguna dan mencegah mereka memilih tanggal yang tidak valid atau di luar jangkauan.
* **Tampilkan Tanggal yang Relevan**: Gunakan `initialDate` untuk menampilkan tanggal yang relevan, misalnya hari ini atau tanggal yang sudah tersimpan sebelumnya.
* **Tunggu Hasil**: Karena `showDatePicker` adalah fungsi asinkron, gunakan `async` dan `await` untuk menangani nilai yang dikembalikan. Nilai yang dikembalikan bisa `null` jika pengguna menutup dialog tanpa memilih tanggal.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const DatePickerExample());

class DatePickerExample extends StatefulWidget {
  const DatePickerExample({Key? key}) : super(key: key);

  @override
  State<DatePickerExample> createState() => _DatePickerExampleState();
}

class _DatePickerExampleState extends State<DatePickerExample> {
  DateTime? _selectedDate;

  Future<void> _showDatePickerDialog() async {
    final DateTime? pickedDate = await showDatePicker(
      context: context,
      initialDate: _selectedDate ?? DateTime.now(),
      firstDate: DateTime(2000),
      lastDate: DateTime(2101),
    );
    if (pickedDate != null && pickedDate != _selectedDate) {
      setState(() {
        _selectedDate = pickedDate;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('DatePicker Contoh')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                _selectedDate == null
                    ? 'Tidak ada tanggal yang dipilih'
                    : 'Tanggal dipilih: ${_selectedDate!.toLocal().toString().split(' ')[0]}',
                style: const TextStyle(fontSize: 18),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _showDatePickerDialog,
                child: const Text('Pilih Tanggal'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}