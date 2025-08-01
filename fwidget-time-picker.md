# [flutter] Widget - TimePicker: Time Picker

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`showTimePicker` adalah fungsi yang menampilkan dialog pop-up yang memungkinkan pengguna untuk memilih satu waktu. Dialog ini menampilkan jam dan menit dalam format yang mudah digunakan.

Lihat dokumentasi resmi Flutter di [https://api.flutter.dev/flutter/material/showTimePicker.html](https://api.flutter.dev/flutter/material/showTimePicker.html).

## Methods and Parameters
---
Untuk menampilkan dialog `TimePicker`, Anda menggunakan fungsi **`showTimePicker`**. Parameter pentingnya adalah:
* **`context`**: Konteks build dari widget saat ini.
* **`initialTime`**: Waktu yang akan ditampilkan saat dialog pertama kali dibuka. Menggunakan `TimeOfDay.now()` adalah praktik umum.

Fungsi ini juga mengembalikan `Future<TimeOfDay?>`, yang berarti Anda harus menggunakan `await` untuk mendapatkan waktu yang dipilih.

## Best Practices or Tips
---
* **Gunakan Waktu Saat Ini**: Atur `initialTime` ke `TimeOfDay.now()` untuk memberikan waktu default yang relevan bagi pengguna.
* **Asinkron**: Gunakan `async` dan `await` untuk menangani nilai yang dikembalikan. `TimeOfDay` berisi jam dan menit, yang dapat Anda gunakan untuk memperbarui status.
* **Perhatikan Format Waktu**: Di beberapa perangkat, format waktu bisa 12 jam atau 24 jam. UI `TimePicker` akan menyesuaikan dengan pengaturan perangkat, jadi pastikan kode Anda dapat menangani keduanya.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const TimePickerExample());

class TimePickerExample extends StatefulWidget {
  const TimePickerExample({Key? key}) : super(key: key);

  @override
  State<TimePickerExample> createState() => _TimePickerExampleState();
}

class _TimePickerExampleState extends State<TimePickerExample> {
  TimeOfDay? _selectedTime;

  Future<void> _showTimePicker() async {
    final TimeOfDay? pickedTime = await showTimePicker(
      context: context,
      initialTime: _selectedTime ?? TimeOfDay.now(),
    );
    if (pickedTime != null && pickedTime != _selectedTime) {
      setState(() {
        _selectedTime = pickedTime;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('TimePicker Contoh')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                _selectedTime == null
                    ? 'Tidak ada waktu yang dipilih'
                    : 'Waktu dipilih: ${_selectedTime!.format(context)}',
                style: const TextStyle(fontSize: 18),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _showTimePicker,
                child: const Text('Pilih Waktu'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}