# [flutter] Widget - DateRangePicker: Date Range Picker

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`showDateRangePicker` adalah fungsi yang menampilkan dialog pop-up yang memungkinkan pengguna untuk memilih rentang tanggal, yaitu tanggal mulai dan tanggal akhir. Ini sering digunakan untuk fitur seperti pemesanan liburan atau pemilihan periode laporan.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/material/showDateRangePicker.html).

## Methods and Parameters
---
Fungsi **`showDateRangePicker`** memiliki parameter yang mirip dengan `showDatePicker`, tetapi dikhususkan untuk rentang:
* **`context`**: Konteks build dari widget saat ini.
* **`firstDate`**: Tanggal paling awal yang dapat dipilih.
* **`lastDate`**: Tanggal paling akhir yang dapat dipilih.
* **`initialDateRange`** (opsional): Objek `DateTimeRange` yang mendefinisikan rentang tanggal yang akan ditampilkan saat dialog pertama kali dibuka.

Fungsi ini mengembalikan `Future<DateTimeRange?>`.

## Best Practices or Tips
---
* **Atur Rentang yang Jelas**: Sama seperti `DatePicker`, penting untuk mengatur `firstDate` dan `lastDate` agar pengguna tidak bisa memilih tanggal di luar rentang yang valid.
* **Tampilkan Rentang Sebelumnya**: Jika ada rentang tanggal yang sudah dipilih sebelumnya, gunakan `initialDateRange` untuk mempermudah pengguna.
* **Validasi Hasil**: Nilai yang dikembalikan adalah `DateTimeRange?`. Pastikan untuk memeriksa apakah nilai tersebut bukan `null` sebelum digunakan, karena pengguna bisa saja menutup dialog.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const DateRangePickerExample());

class DateRangePickerExample extends StatefulWidget {
  const DateRangePickerExample({Key? key}) : super(key: key);

  @override
  State<DateRangePickerExample> createState() => _DateRangePickerExampleState();
}

class _DateRangePickerExampleState extends State<DateRangePickerExample> {
  DateTimeRange? _selectedRange;

  Future<void> _showDateRangePicker() async {
    final DateTimeRange? pickedRange = await showDateRangePicker(
      context: context,
      firstDate: DateTime(2020),
      lastDate: DateTime(2030),
      initialDateRange: _selectedRange,
    );
    if (pickedRange != null && pickedRange != _selectedRange) {
      setState(() {
        _selectedRange = pickedRange;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('DateRangePicker Contoh')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                _selectedRange == null
                    ? 'Tidak ada rentang tanggal yang dipilih'
                    : 'Rentang dipilih: ${_selectedRange!.start.toLocal().toString().split(' ')[0]} - ${_selectedRange!.end.toLocal().toString().split(' ')[0]}',
                style: const TextStyle(fontSize: 18),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _showDateRangePicker,
                child: const Text('Pilih Rentang Tanggal'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}