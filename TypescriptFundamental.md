# [typescript] #01 - TS Introduction

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## Apa itu TypeScript?
- TypeScript adalah bahasa pemrograman yang **dibangun di atas JavaScript**.
- Bersifat **strongly typed** dan **statically typed** (_static programming language_).
- Semua kode JavaScript valid juga bisa dijalankan di TypeScript.

## Tujuan TypeScript
- Menambahkan **sistem tipe statis** ke JavaScript.
- Membuat pengembangan aplikasi besar jadi **lebih aman, terstruktur, dan scalable**.
- Mendeteksi **kesalahan tipe sejak tahap kompilasi**, bukan saat runtime.

## Keuntungan Menggunakan TypeScript
- Mendapatkan **autocomplete**, **navigasi kode**, dan **refactor** yang lebih akurat.
- Menurunkan risiko bug karena kesalahan tipe data.
- Memudahkan kerja tim dan pemeliharaan kode jangka panjang.
- **100% kompatibel dengan JavaScript**, bisa diadopsi secara bertahap.

## Cara Kerja TypeScript
- Kamu menulis kode di file `.ts` menggunakan JavaScript + tipe statis.
- TypeScript dikompilasi (transpile) ke JavaScript menggunakan `tsc`.
- Saat kompilasi, TypeScript mengecek tipe dan memberi peringatan/error sebelum dijalankan.
- Hasil kompilasi adalah file `.js` yang bisa dijalankan di browser atau Node.js.
- Editor seperti VS Code menggunakan info tipe dari TypeScript untuk fitur seperti autocomplete dan refactor.
- Pengaturan proyek TypeScript dikelola lewat file `tsconfig.json`.
