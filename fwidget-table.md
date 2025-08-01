# [flutter] Widget - Table: Tabular Layout

![widget](https://raw.githubusercontent.com/oujisan/OuVault/main/img/flutter-widget.png)

## Description
---
`Table` adalah widget yang mengatur widget anaknya dalam tata letak tabel yang rapi, dengan baris (row) dan kolom (column). Ini berguna untuk menampilkan data terstruktur dalam format grid yang sederhana.

Lihat dokumentasi resmi Flutter di [docs](https://api.flutter.dev/flutter/widgets/Table-class.html).

## Methods and Parameters
---
`Table` memiliki parameter berikut untuk mengatur tata letaknya:
* **`children`**: Daftar `TableRow`, yang setiap `TableRow` berisi daftar widget anak (`List<Widget>`) untuk setiap sel di baris tersebut.
* **`columnWidths`** (opsional): `Map<int, TableColumnWidth>` yang menentukan lebar untuk setiap kolom. Anda dapat menggunakan `FlexColumnWidth` (lebar fleksibel) atau `FixedColumnWidth` (lebar tetap).
* **`border`** (opsional): Mengatur bingkai (`TableBorder`) untuk tabel, seperti garis batas di antara sel.

## Best Practices or Tips
---
* **Konsistensi Kolom**: Pastikan setiap `TableRow` memiliki jumlah widget anak yang sama dengan jumlah kolom yang Anda harapkan.
* **Kontrol Lebar**: Gunakan `TableColumnWidth` untuk mengontrol bagaimana setiap kolom menyesuaikan ukurannya. `FlexColumnWidth` sangat berguna untuk membuat tata letak yang responsif.
* **Alternatif untuk Tabel Kompleks**: Jika Anda membutuhkan tata letak grid yang lebih dinamis dan dapat digulir, pertimbangkan `GridView`. Untuk data yang sangat banyak, `ListView` dengan `ListTile` mungkin lebih baik.

## Example
---
```dart
import 'package:flutter/material.dart';

void main() => runApp(const TableExample());

class TableExample extends StatelessWidget {
  const TableExample({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Table Contoh')),
        body: Center(
          child: Table(
            border: TableBorder.all(color: Colors.grey),
            columnWidths: const <int, TableColumnWidth>{
              0: FlexColumnWidth(1),
              1: FlexColumnWidth(3),
              2: FlexColumnWidth(1),
            },
            children: const <TableRow>[
              TableRow(
                children: <Widget>[
                  TableCell(child: Center(child: Text('No.'))),
                  TableCell(child: Center(child: Text('Nama'))),
                  TableCell(child: Center(child: Text('Usia'))),
                ],
              ),
              TableRow(
                children: <Widget>[
                  TableCell(child: Center(child: Text('1'))),
                  TableCell(child: Center(child: Text('John Doe'))),
                  TableCell(child: Center(child: Text('30'))),
                ],
              ),
              TableRow(
                children: <Widget>[
                  TableCell(child: Center(child: Text('2'))),
                  TableCell(child: Center(child: Text('Jane Smith'))),
                  TableCell(child: Center(child: Text('25'))),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}