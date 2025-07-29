# [typescript] #02 - TS Basic Types

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## Primitive Types

TypeScript mendukung semua tipe primitif JavaScript dengan tambahan type annotation.

### String

```typescript
let message: string = "Hello TypeScript";
let template: string = `Welcome ${message}`;

// String methods tetap tersedia
let upperCase: string = message.toUpperCase();
```

### Number

```typescript
let integer: number = 42;
let float: number = 3.14;
let binary: number = 0b1010; // Binary
let octal: number = 0o744;   // Octal
let hex: number = 0xff;      // Hexadecimal
```

### Boolean

```typescript
let isComplete: boolean = true;
let isLoading: boolean = false;

// Dari hasil evaluasi
let hasData: boolean = data.length > 0;
```

### Null & Undefined

```typescript
let nullValue: null = null;
let undefinedValue: undefined = undefined;

// Dalam strict mode, harus eksplisit
let optionalString: string | null = null;
let maybeNumber: number | undefined = undefined;
```

## Array Types

### Basic Array Syntax

```typescript
// Dua cara penulisan array
let numbers: number[] = [1, 2, 3, 4, 5];
let strings: Array<string> = ["hello", "world"];

// Mixed types dengan union
let mixed: (string | number)[] = ["hello", 42, "world"];
```

### Array Methods

```typescript
let fruits: string[] = ["apple", "banana", "orange"];

// Type-safe array operations
fruits.push("grape");        // ✓ Valid
// fruits.push(123);         // ✗ Error: number tidak bisa di-push ke string[]

let firstFruit: string = fruits[0];
let length: number = fruits.length;
```

## Object Types

### Object Literal Types

```typescript
// Inline object type
let person: { name: string; age: number; email?: string } = {
    name: "John Doe",
    age: 30
    // email optional
};

// Nested objects
let address: {
    street: string;
    city: string;
    coordinates: { lat: number; lng: number };
} = {
    street: "123 Main St",
    city: "Jakarta",
    coordinates: { lat: -6.2088, lng: 106.8456 }
};
```

### Optional Properties

```typescript
let user: {
    id: number;
    name: string;
    email?: string;  // Optional property
    phone?: string;  // Optional property
} = {
    id: 1,
    name: "Alice"
    // email dan phone tidak wajib
};
```

## Function Types

### Function Declarations

```typescript
// Function dengan parameter dan return type
function add(x: number, y: number): number {
    return x + y;
}

// Function tanpa return value
function logMessage(message: string): void {
    console.log(message);
}

// Optional parameters
function greet(name: string, greeting?: string): string {
    return `${greeting || "Hello"}, ${name}!`;
}

// Default parameters
function createUser(name: string, role: string = "user"): object {
    return { name, role };
}
```

### Arrow Functions

```typescript
// Arrow function dengan explicit types
const multiply = (x: number, y: number): number => x * y;

// Arrow function dengan inferred return type
const divide = (x: number, y: number) => x / y; // return type: number

// Function sebagai variable
let calculator: (x: number, y: number) => number;
calculator = multiply; // ✓ Valid
calculator = divide;   // ✓ Valid
```

## Union Types

Union types memungkinkan variable memiliki lebih dari satu tipe.

```typescript
// Basic union
let id: string | number;
id = "USER123";  // ✓ Valid
id = 456;        // ✓ Valid
// id = true;    // ✗ Error

// Union dengan literal types
let status: "loading" | "success" | "error" = "loading";

// Function dengan union parameters
function printId(id: string | number): void {
    // Type narrowing diperlukan
    if (typeof id === "string") {
        console.log(`ID: ${id.toUpperCase()}`);
    } else {
        console.log(`ID: ${id.toFixed(0)}`);
    }
}
```

## Type Aliases

Type aliases membantu membuat kode lebih readable dan reusable.

```typescript
// Basic type alias
type StringOrNumber = string | number;
type UserID = StringOrNumber;

// Object type alias
type User = {
    id: UserID;
    name: string;
    email: string;
    isActive: boolean;
};

// Function type alias
type EventHandler = (event: string) => void;

// Using type aliases
let currentUser: User = {
    id: "USER123",
    name: "John Doe",
    email: "john@example.com",
    isActive: true
};

let handleClick: EventHandler = (event) => {
    console.log(`Handling ${event}`);
};
```

## Literal Types

Literal types memungkinkan kita menentukan nilai eksak yang bisa diterima.

```typescript
// String literals
let direction: "north" | "south" | "east" | "west" = "north";

// Number literals
let dice: 1 | 2 | 3 | 4 | 5 | 6 = 6;

// Boolean literals (jarang digunakan)
let isTrue: true = true;

// Combining with other types
type ButtonSize = "small" | "medium" | "large";
type ButtonVariant = "primary" | "secondary" | "danger";

interface Button {
    text: string;
    size: ButtonSize;
    variant: ButtonVariant;
    disabled?: boolean;
}
```

Basic types ini adalah fondasi dari sistem tipe TypeScript. Dengan memahami ini, kita bisa membangun aplikasi yang lebih robust dan menangkap error lebih awal dalam proses development.