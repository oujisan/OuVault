# [typescript] #05 - TS Generics

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## What are Generics?

Generics memungkinkan kita membuat komponen yang bisa bekerja dengan berbagai tipe data sambil tetap mempertahankan type safety. Dengan generics, kita bisa menulis kode yang reusable dan flexible tanpa kehilangan informasi tipe.

## Basic Generic Functions

```typescript
// Function tanpa generics - tidak flexible
function identityString(arg: string): string {
    return arg;
}

function identityNumber(arg: number): number {
    return arg;
}

// Function dengan generics - flexible dan type-safe
function identity<T>(arg: T): T {
    return arg;
}

// Usage dengan explicit type
let stringResult = identity<string>("hello");
let numberResult = identity<number>(42);
let booleanResult = identity<boolean>(true);

// Type inference - TypeScript otomatis detect type
let autoString = identity("hello");    // T inferred as string
let autoNumber = identity(42);         // T inferre
```