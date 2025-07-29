# [typescript] #01 - TS Introduction & Setup

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## What is TypeScript?

TypeScript adalah bahasa pemrograman yang dikembangkan oleh Microsoft sebagai superset dari JavaScript. Artinya, semua kode JavaScript yang valid juga merupakan kode TypeScript yang valid. TypeScript menambahkan sistem tipe statis opsional ke JavaScript, yang membantu mendeteksi error sejak dini dalam proses development.

### Key Benefits

- **Static Type Checking**: Mendeteksi error sebelum runtime
- **Better IDE Support**: Autocomplete, refactoring, dan navigation yang lebih baik
- **Enhanced Code Documentation**: Tipe berfungsi sebagai dokumentasi hidup
- **Easier Refactoring**: Perubahan kode besar menjadi lebih aman
- **Modern JavaScript Features**: Mendukung fitur ES6+ bahkan untuk target yang lebih lama

## Installation & Setup

### Installing TypeScript

```typescript
// Global installation
npm install -g typescript

// Project-specific installation
npm install --save-dev typescript
```

### Creating tsconfig.json

```typescript
// Initialize TypeScript project
tsc --init

// Or create manually
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### Basic Compilation

```typescript
// Compile single file
tsc app.ts

// Compile with config
tsc

// Watch mode
tsc --watch
```

## Your First TypeScript File

```typescript
// hello.ts
function greet(name: string): string {
    return `Hello, ${name}!`;
}

const userName: string = "World";
console.log(greet(userName));

// Error: Argument of type 'number' is not assignable to parameter of type 'string'
// greet(123);
```

TypeScript compiler akan menghasilkan JavaScript yang bersih dan readable. File di atas akan dikompilasi menjadi:

```typescript
// hello.js
function greet(name) {
    return "Hello, " + name + "!";
}
var userName = "World";
console.log(greet(userName));
```

## Development Workflow

### Recommended Setup

1. **Code Editor**: Visual Studio Code dengan TypeScript extension
2. **Build Tool**: Webpack, Vite, atau Parcel dengan TypeScript loader
3. **Testing**: Jest dengan @types/jest
4. **Linting**: ESLint dengan @typescript-eslint

### Package.json Scripts

```typescript
{
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch",
    "start": "node dist/index.js"
  }
}
```

TypeScript memungkinkan kita menulis JavaScript yang lebih robust dan maintainable dengan memberikan confidence bahwa kode kita akan bekerja seperti yang diharapkan.