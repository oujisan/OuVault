# [typescript] #03 - TS Interfaces

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## What are Interfaces?

Interface adalah cara untuk mendefinisikan struktur object dalam TypeScript. Berbeda dengan type aliases, interface lebih fokus pada shape of objects dan bisa di-extend. Interface sangat berguna untuk mendefinisikan kontrak yang harus dipenuhi oleh object atau class.

## Basic Interface

```typescript
// Interface definition
interface User {
    id: number;
    name: string;
    email: string;
    isActive: boolean;
}

// Menggunakan interface
const user: User = {
    id: 1,
    name: "John Doe",
    email: "john@example.com",
    isActive: true
};

// Function yang menggunakan interface
function displayUser(user: User): void {
    console.log(`${user.name} (${user.email})`);
}
```

## Optional Properties

Property yang tidak wajib ada bisa ditandai dengan `?`.

```typescript
interface Product {
    id: number;
    name: string;
    price: number;
    description?: string;  // Optional
    category?: string;     // Optional
    discount?: number;     // Optional
}

// Valid tanpa optional properties
const product1: Product = {
    id: 1,
    name: "Laptop",
    price: 15000000
};

// Valid dengan optional properties
const product2: Product = {
    id: 2,
    name: "Mouse",
    price: 200000,
    description: "Wireless gaming mouse",
    category: "Electronics"
};
```

## Readonly Properties

Property yang tidak bisa diubah setelah object dibuat.

```typescript
interface Config {
    readonly apiUrl: string;
    readonly version: string;
    timeout: number;  // Bisa diubah
}

const appConfig: Config = {
    apiUrl: "https://api.example.com",
    version: "1.0.0",
    timeout: 5000
};

// appConfig.apiUrl = "https://new-api.com"; // ✗ Error: readonly
appConfig.timeout = 10000; // ✓ Valid
```

## Function Properties

Interface bisa mendefinisikan method atau function properties.

```typescript
interface Calculator {
    // Method signatures
    add(x: number, y: number): number;
    subtract(x: number, y: number): number;
    
    // Function property
    multiply: (x: number, y: number) => number;
    
    // Optional method
    divide?(x: number, y: number): number;
}

const calculator: Calculator = {
    add(x, y) {
        return x + y;
    },
    subtract(x, y) {
        return x - y;
    },
    multiply: (x, y) => x * y
    // divide tidak wajib karena optional
};
```

## Index Signatures

Untuk object dengan property names yang dinamis.

```typescript
// String index signature
interface StringDictionary {
    [key: string]: string;
}

const translations: StringDictionary = {
    hello: "Halo",
    goodbye: "Selamat tinggal",
    thanks: "Terima kasih"
};

// Number index signature
interface NumberArray {
    [index: number]: number;
}

const scores: NumberArray = {
    0: 85,
    1: 92,
    2: 78
};

// Mixed index signatures
interface FlexibleObject {
    // Properties yang pasti ada
    name: string;
    id: number;
    
    // Dynamic properties
    [key: string]: any;
}
```

## Extending Interfaces

Interface bisa meng-extend interface lain untuk reusability.

```typescript
// Base interface
interface Animal {
    name: string;
    age: number;
}

// Extended interface
interface Dog extends Animal {
    breed: string;
    isVaccinated: boolean;
}

// Multiple inheritance
interface Pet {
    owner: string;
}

interface ServiceDog extends Dog, Pet {
    serviceType: string;
}

const myDog: ServiceDog = {
    name: "Buddy",
    age: 3,
    breed: "Golden Retriever",
    isVaccinated: true,
    owner: "John Smith",
    serviceType: "Guide Dog"
};
```

## Interface vs Type Aliases

Meskipun mirip, ada perbedaan penting antara interface dan type aliases.

```typescript
// Interface - bisa di-extend dan di-merge
interface UserInterface {
    name: string;
}

interface UserInterface {
    email: string;  // Declaration merging
}

// Type alias - tidak bisa di-merge
type UserType = {
    name: string;
};

// Extending dengan intersection
type ExtendedUser = UserType & {
    email: string;
};

// Interface untuk object shapes
interface ApiResponse {
    data: any;
    status: number;
    message: string;
}

// Type alias untuk unions dan complex types
type Status = "loading" | "success" | "error";
type ID = string | number;
```

## Generic Interfaces

Interface bisa menggunakan generics untuk fleksibilitas yang lebih besar.

```typescript
// Generic interface
interface Repository<T> {
    items: T[];
    add(item: T): void;
    findById(id: number): T | undefined;
    update(id: number, item: Partial<T>): void;
    delete(id: number): boolean;
}

// Usage with specific types
interface Book {
    id: number;
    title: string;
    author: string;
}

class BookRepository implements Repository<Book> {
    items: Book[] = [];
    
    add(book: Book): void {
        this.items.push(book);
    }
    
    findById(id: number): Book | undefined {
        return this.items.find(book => book.id === id);
    }
    
    update(id: number, updates: Partial<Book>): void {
        const book = this.findById(id);
        if (book) {
            Object.assign(book, updates);
        }
    }
    
    delete(id: number): boolean {
        const index = this.items.findIndex(book => book.id === id);
        if (index > -1) {
            this.items.splice(index, 1);
            return true;
        }
        return false;
    }
}
```

## Interface for Classes

Interface bisa digunakan sebagai kontrak yang harus diimplementasikan oleh class.

```typescript
interface Flyable {
    altitude: number;
    fly(): void;
    land(): void;
}

interface Swimmable {
    depth: number;
    swim(): void;
    dive(depth: number): void;
}

// Class implementing single interface
class Bird implements Flyable {
    altitude: number = 0;
    
    fly(): void {
        this.altitude = 1000;
        console.log("Bird is flying");
    }
    
    land(): void {
        this.altitude = 0;
        console.log("Bird has landed");
    }
}

// Class implementing multiple interfaces
class Duck implements Flyable, Swimmable {
    altitude: number = 0;
    depth: number = 0;
    
    fly(): void {
        this.altitude = 500;
        console.log("Duck is flying");
    }
    
    land(): void {
        this.altitude = 0;
        console.log("Duck has landed");
    }
    
    swim(): void {
        console.log("Duck is swimming");
    }
    
    dive(depth: number): void {
        this.depth = depth;
        console.log(`Duck diving to ${depth}m`);
    }
}
```

Interface adalah alat yang powerful untuk mendefinisikan struktur data dan kontrak dalam TypeScript. Mereka membantu membuat kode yang lebih predictable, maintainable, dan self-documenting.