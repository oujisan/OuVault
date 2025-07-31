# [typescript] #09 - TS Configuration & Tooling

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## TSConfig.json

File konfigurasi utama TypeScript yang mengontrol bagaimana TypeScript compiler bekerja.

### Basic Configuration

```typescript
// tsconfig.json - Basic setup
{
  "compilerOptions": {
    // Target ECMAScript version
    "target": "ES2020",
    
    // Module system
    "module": "commonjs",
    
    // Output directory
    "outDir": "./dist",
    
    // Source directory
    "rootDir": "./src",
    
    // Enable all strict type checking
    "strict": true,
    
    // Enable ES module interop
    "esModuleInterop": true,
    
    // Skip lib checking for faster compilation
    "skipLibCheck": true,
    
    // Force consistent casing
    "forceConsistentCasingInFileNames": true
  },
  
  // Files to include
  "include": [
    "src/**/*"
  ],
  
  // Files to exclude
  "exclude": [
    "node_modules",
    "dist",
    "**/*.test.ts"
  ]
}
```

### Advanced Configuration

```typescript
// tsconfig.json - Advanced setup
{
  "compilerOptions": {
    // Language and environment
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "moduleResolution": "node",
    
    // Output configuration
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,          // Generate .d.ts files
    "declarationMap": true,       // Generate .d.ts.map files
    "sourceMap": true,           // Generate .js.map files
    "removeComments": false,     // Keep comments in output
    
    // Path mapping
    "baseUrl": "./",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"],
      "@services/*": ["src/services/*"]
    },
    
    // Type checking
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true,
    "strictPropertyInitialization": true,
    "noImplicitReturns": true,
    "noImplicitThis": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "exactOptionalPropertyTypes": true,
    
    // Module resolution
    "allowSyntheticDefaultImports": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    
    // Emit
    "noEmit": false,
    "emitDecoratorMetadata": true,
    "experimentalDecorators": true,
    
    // Advanced
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "useDefineForClassFields": true,
    "allowUnusedLabels": false,
    "allowUnreachableCode": false,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true
  },
  
  "include": [
    "src/**/*",
    "types/**/*"
  ],
  
  "exclude": [
    "node_modules",
    "dist",
    "build",
    "**/*.test.ts",
    "**/*.spec.ts",
    "coverage"
  ],
  
  // Type acquisition for JavaScript files
  "typeAcquisition": {
    "enable": true,
    "include": ["jest", "node"],
    "exclude": ["jquery"]
  }
}
```

### Project References

Untuk monorepo atau large projects dengan multiple packages.

```typescript
// Root tsconfig.json
{
  "files": [],
  "references": [
    { "path": "./packages/shared" },
    { "path": "./packages/client" },
    { "path": "./packages/server" }
  ]
}

// packages/shared/tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "declaration": true,
    "composite": true    // Enable project references
  },
  "include": ["src/**/*"]
}

// packages/client/tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "lib": ["DOM", "ES2020"]
  },
  "references": [
    { "path": "../shared" }
  ],
  "include": ["src/**/*"]
}

// Build command for project references
// tsc --build
// tsc --build --watch
```

## Compiler Options Deep Dive

### Type Checking Options

```typescript
{
  "compilerOptions": {
    // Strict mode - enables all strict options
    "strict": true,
    
    // Individual strict options
    "noImplicitAny": true,              // Error on 'any' type
    "strictNullChecks": true,           // Null/undefined checking
    "strictFunctionTypes": true,        // Function type checking
    "strictBindCallApply": true,        // bind/call/apply checking
    "strictPropertyInitialization": true, // Class property initialization
    "noImplicitReturns": true,          // Error on missing return
    "noImplicitThis": true,             // Error on 'this' with 'any' type
    "noUnusedLocals": true,             // Error on unused variables
    "noUnusedParameters": true,         // Error on unused parameters
    "exactOptionalPropertyTypes": true, // Exact optional property types
    "noUncheckedIndexedAccess": true,   // Add undefined to index signatures
    
    // Additional checks
    "noFallthroughCasesInSwitch": true, // Error on fallthrough cases
    "noImplicitOverride": true,         // Require 'override' keyword
    "allowUnusedLabels": false,         // Error on unused labels
    "allowUnreachableCode": false       // Error on unreachable code
  }
}

// Example of strict null checks
function processUser(user: User | null) {
    // With strictNullChecks: true
    // console.log(user.name); // ✗ Error: Object is possibly null
    
    if (user !== null) {
        console.log(user.name); // ✓ Valid - null check performed
    }
}
```

### Module Options

```typescript
{
  "compilerOptions": {
    // Module system
    "module": "ESNext",              // ES modules
    // "module": "CommonJS",         // Node.js modules
    // "module": "AMD",              // AMD modules
    // "module": "UMD",              // Universal modules
    
    // Module resolution
    "moduleResolution": "node",      // Node.js resolution
    // "moduleResolution": "classic", // TypeScript classic resolution
    
    // Allow importing JSON files
    "resolveJsonModule": true,
    
    // ES module interop
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    
    // Isolated modules (for tools like Babel)
    "isolatedModules": true
  }
}

// Example usage
import config from './config.json';  // With resolveJsonModule
import React from 'react';           // With allowSyntheticDefaultImports
```

### Emit Options

```typescript
{
  "compilerOptions": {
    // Output options
    "outDir": "./dist",              // Output directory
    "outFile": "./bundle.js",        // Single output file (AMD/System only)
    "rootDir": "./src",              // Root source directory
    
    // Declaration files
    "declaration": true,             // Generate .d.ts files
    "declarationDir": "./types",     // Separate directory for declarations
    "declarationMap": true,          // Generate .d.ts.map files
    "emitDeclarationOnly": false,    // Only emit declarations
    
    // Source maps
    "sourceMap": true,               // Generate .js.map files
    "inlineSourceMap": false,        // Inline source maps
    "sourceRoot": "",                // Source root for debugger
    
    // Comments and formatting
    "removeComments": false,         // Remove comments from output
    "newLine": "lf",                 // Line ending style
    "preserveConstEnums": true,      // Don't erase const enum declarations
    
    // Emit control
    "noEmit": false,                 // Don't emit output
    "noEmitOnError": true,           // Don't emit if there are errors
    "noEmitHelpers": false,          // Don't generate helper functions
    "importHelpers": false,          // Import helpers from tslib
    
    // Down-level iteration
    "downlevelIteration": true       // Emit more compliant iteration code
  }
}
```

## ESLint Integration

Combining TypeScript with ESLint for comprehensive code quality.

### Installation

```typescript
// Package installation
npm install --save-dev @typescript-eslint/parser @typescript-eslint/eslint-plugin eslint

// .eslintrc.js
module.exports = {
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    project: './tsconfig.json',
  },
  plugins: ['@typescript-eslint'],
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    '@typescript-eslint/recommended-requiring-type-checking',
  ],
  rules: {
    // TypeScript specific rules
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/prefer-const': 'error',
    '@typescript-eslint/no-inferrable-types': 'off',
    '@typescript-eslint/explicit-function-return-type': 'warn',
    '@typescript-eslint/explicit-module-boundary-types': 'warn',
    '@typescript-eslint/no-non-null-assertion': 'warn',
    '@typescript-eslint/prefer-optional-chain': 'error',
    '@typescript-eslint/prefer-nullish-coalescing': 'error',
    
    // General rules
    'no-console': 'warn',
    'prefer-const': 'error',
    'no-var': 'error',
    'object-shorthand': 'error',
    'prefer-arrow-callback': 'error',
  },
  env: {
    node: true,
    browser: true,
    es6: true,
  },
};
```

### Prettier Integration

```typescript
// .prettierrc
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true,
  "bracketSameLine": false,
  "arrowParens": "avoid"
}

// .eslintrc.js - Add Prettier
module.exports = {
  // ... other config
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'prettier', // Must be last
  ],
  plugins: ['@typescript-eslint', 'prettier'],
  rules: {
    // ... other rules
    'prettier/prettier': 'error',
  },
};

// Package installation
npm install --save-dev prettier eslint-config-prettier eslint-plugin-prettier
```

## Build Tools Integration

### Webpack Configuration

```typescript
// webpack.config.js
const path = require('path');

module.exports = {
  entry: './src/index.ts',
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@components': path.resolve(__dirname, 'src/components'),
      '@utils': path.resolve(__dirname, 'src/utils'),
    },
  },
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  devtool: 'source-map',
  mode: 'development',
};

// Alternative: using fork-ts-checker-webpack-plugin
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');

module.exports = {
  // ... other config
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: {
          loader: 'ts-loader',
          options: {
            transpileOnly: true, // Let fork-ts-checker handle type checking
          },
        },
        exclude: /node_modules/,
      },
    ],
  },
  plugins: [
    new ForkTsCheckerWebpackPlugin({
      typescript: {
        configFile: path.resolve(__dirname, 'tsconfig.json'),
      },
    }),
  ],
};
```

### Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'MyLib',
      fileName: 'my-lib',
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@utils': resolve(__dirname, 'src/utils'),
    },
  },
  esbuild: {
    target: 'es2020',
  },
});

// For React projects
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
});
```

### Rollup Configuration

```typescript
// rollup.config.js
import typescript from '@rollup/plugin-typescript';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import { terser } from 'rollup-plugin-terser';

export default {
  input: 'src/index.ts',
  output: [
    {
      file: 'dist/bundle.cjs.js',
      format: 'cjs',
      sourcemap: true,
    },
    {
      file: 'dist/bundle.esm.js',
      format: 'esm',
      sourcemap: true,
    },
    {
      file: 'dist/bundle.umd.js',
      format: 'umd',
      name: 'MyLibrary',
      sourcemap: true,
    },
  ],
  plugins: [
    nodeResolve(),
    commonjs(),
    typescript({
      tsconfig: './tsconfig.json',
      declaration: true,
      declarationDir: 'dist/types',
    }),
    terser(), // Minification
  ],
  external: ['react', 'react-dom'], // Don't bundle these
};
```

## Testing Configuration

### Jest with TypeScript

```typescript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: [
    '**/__tests__/**/*.+(ts|tsx|js)',
    '**/*.(test|spec).+(ts|tsx|js)',
  ],
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.ts',
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@components/(.*)$': '<rootDir>/src/components/$1',
    '^@utils/(.*)$': '<rootDir>/src/utils/$1',
  },
};

// tsconfig.json for tests
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "types": ["jest", "node"]
  },
  "include": [
    "src/**/*",
    "**/*.test.ts",
    "**/*.spec.ts"
  ]
}

// Example test file
// src/utils/math.test.ts
import { add, multiply } from './math';

describe('Math utilities', () => {
  describe('add', () => {
    it('should add two numbers correctly', () => {
      expect(add(2, 3)).toBe(5);
      expect(add(-1, 1)).toBe(0);
    });
  });

  describe('multiply', () => {
    it('should multiply two numbers correctly', () => {
      expect(multiply(3, 4)).toBe(12);
      expect(multiply(0, 5)).toBe(0);
    });
  });
});
```

## Package.json Scripts

```typescript
// package.json
{
  "name": "my-typescript-project",
  "version": "1.0.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    // Development
    "dev": "tsc --watch",
    "dev:node": "ts-node src/index.ts",
    "dev:nodemon": "nodemon --exec ts-node src/index.ts",
    
    // Building
    "build": "tsc",
    "build:prod": "tsc --project tsconfig.prod.json",
    "build:clean": "rimraf dist && npm run build",
    "build:watch": "tsc --watch",
    
    // Type checking
    "type-check": "tsc --noEmit",
    "type-check:watch": "tsc --noEmit --watch",
    
    // Linting and formatting
    "lint": "eslint src/**/*.{ts,tsx}",
    "lint:fix": "eslint src/**/*.{ts,tsx} --fix",
    "format": "prettier --write src/**/*.{ts,tsx,json,md}",
    "format:check": "prettier --check src/**/*.{ts,tsx,json,md}",
    
    // Testing
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:ci": "jest --ci --coverage --watchAll=false",
    
    // Pre-commit hooks
    "pre-commit": "lint-staged",
    "prepare": "husky install",
    
    // Utilities
    "clean": "rimraf dist coverage .nyc_output",
    "start": "node dist/index.js",
    "start:prod": "NODE_ENV=production node dist/index.js"
  },
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md}": [
      "prettier --write"
    ]
  },
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged",
      "pre-push": "npm run type-check && npm run test"
    }
  },
  "devDependencies": {
    "@types/node": "^18.0.0",
    "@typescript-eslint/eslint-plugin": "^5.30.0",
    "@typescript-eslint/parser": "^5.30.0",
    "eslint": "^8.18.0",
    "eslint-config-prettier": "^8.5.0",
    "eslint-plugin-prettier": "^4.2.1",
    "husky": "^8.0.1",
    "jest": "^28.1.2",
    "lint-staged": "^13.0.3",
    "nodemon": "^2.0.19",
    "prettier": "^2.7.1",
    "rimraf": "^3.0.2",
    "ts-jest": "^28.0.5",
    "ts-node": "^10.8.2",
    "typescript": "^4.7.4"
  }
}
```

## Environment-Specific Configurations

### Development vs Production

```typescript
// tsconfig.json (base)
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}

// tsconfig.dev.json (development)
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "sourceMap": true,
    "removeComments": false,
    "noEmitOnError": false,
    "incremental": true,
    "tsBuildInfoFile": ".tsbuildinfo"
  },
  "include": [
    "src/**/*",
    "**/*.test.ts",
    "**/*.spec.ts"
  ]
}

// tsconfig.prod.json (production)
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "sourceMap": false,
    "removeComments": true,
    "noEmitOnError": true,
    "declaration": true,
    "declarationMap": false
  },
  "exclude": [
    "node_modules",
    "dist",
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/__tests__/**"
  ]
}

// Build scripts
"scripts": {
  "build:dev": "tsc --project tsconfig.dev.json",
  "build:prod": "tsc --project tsconfig.prod.json"
}
```

### Multiple Entry Points

```typescript
// tsconfig.json untuk library dengan multiple entry points
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "node",
    "declaration": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"]
}

// Package.json dengan multiple exports
{
  "name": "my-library",
  "version": "1.0.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "require": "./dist/index.cjs",
      "types": "./dist/index.d.ts"
    },
    "./utils": {
      "import": "./dist/utils/index.js",
      "require": "./dist/utils/index.cjs",
      "types": "./dist/utils/index.d.ts"
    },
    "./components": {
      "import": "./dist/components/index.js",
      "require": "./dist/components/index.cjs",
      "types": "./dist/components/index.d.ts"
    }
  },
  "files": [
    "dist"
  ]
}
```

## Advanced Tooling

### TypeScript Compiler API

```typescript
// build-script.ts - Custom build script using TS Compiler API
import * as ts from 'typescript';
import * as fs from 'fs';
import * as path from 'path';

function compile(fileNames: string[], options: ts.CompilerOptions): void {
  const program = ts.createProgram(fileNames, options);
  const emitResult = program.emit();

  const allDiagnostics = ts
    .getPreEmitDiagnostics(program)
    .concat(emitResult.diagnostics);

  allDiagnostics.forEach(diagnostic => {
    if (diagnostic.file) {
      const { line, character } = ts.getLineAndCharacterOfPosition(
        diagnostic.file,
        diagnostic.start!
      );
      const message = ts.flattenDiagnosticMessageText(
        diagnostic.messageText,
        '\n'
      );
      console.log(
        `${diagnostic.file.fileName} (${line + 1},${character + 1}): ${message}`
      );
    } else {
      console.log(
        ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')
      );
    }
  });

  const exitCode = emitResult.emitSkipped ? 1 : 0;
  console.log(`Process exiting with code '${exitCode}'.`);
  process.exit(exitCode);
}

// Read tsconfig.json
const configPath = ts.findConfigFile('./', ts.sys.fileExists, 'tsconfig.json');
if (!configPath) {
  throw new Error("Could not find a valid 'tsconfig.json'.");
}

const configFile = ts.readConfigFile(configPath, ts.sys.readFile);
const compilerOptions = ts.parseJsonConfigFileContent(
  configFile.config,
  ts.sys,
  './'
);

compile(compilerOptions.fileNames, compilerOptions.options);
```

### Custom Transformers

```typescript
// custom-transformer.ts
import * as ts from 'typescript';

// Transformer yang menambahkan console.log ke setiap function
const addLoggingTransformer: ts.TransformerFactory<ts.SourceFile> = (
  context: ts.TransformationContext
) => {
  return (sourceFile: ts.SourceFile) => {
    function visit(node: ts.Node): ts.Node {
      if (ts.isFunctionDeclaration(node) && node.name) {
        const logStatement = ts.factory.createExpressionStatement(
          ts.factory.createCallExpression(
            ts.factory.createPropertyAccessExpression(
              ts.factory.createIdentifier('console'),
              ts.factory.createIdentifier('log')
            ),
            undefined,
            [ts.factory.createStringLiteral(`Entering function: ${node.name.text}`)]
          )
        );

        const newBody = ts.factory.createBlock([
          logStatement,
          ...(node.body?.statements || [])
        ]);

        return ts.factory.updateFunctionDeclaration(
          node,
          node.decorators,
          node.modifiers,
          node.asteriskToken,
          node.name,
          node.typeParameters,
          node.parameters,
          node.type,
          newBody
        );
      }

      return ts.visitEachChild(node, visit, context);
    }

    return ts.visitNode(sourceFile, visit);
  };
};

// Usage in compilation
const program = ts.createProgram(['src/index.ts'], {
  target: ts.ScriptTarget.ES2020,
  module: ts.ModuleKind.CommonJS,
});

program.emit(undefined, undefined, undefined, false, {
  before: [addLoggingTransformer],
});
```

### Type-Safe Configuration

```typescript
// config.schema.ts - Type-safe configuration
interface DatabaseConfig {
  host: string;
  port: number;
  username: string;
  password: string;
  database: string;
}

interface ApiConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
}

interface AppConfig {
  env: 'development' | 'staging' | 'production';
  port: number;
  database: DatabaseConfig;
  api: ApiConfig;
  features: {
    enableLogging: boolean;
    enableMetrics: boolean;
    enableAuth: boolean;
  };
}

// config.ts - Configuration loader with validation
import { z } from 'zod';

const DatabaseConfigSchema = z.object({
  host: z.string(),
  port: z.number().min(1).max(65535),
  username: z.string(),
  password: z.string(),
  database: z.string(),
});

const ApiConfigSchema = z.object({
  baseUrl: z.string().url(),
  timeout: z.number().positive(),
  retries: z.number().min(0).max(10),
});

const AppConfigSchema = z.object({
  env: z.enum(['development', 'staging', 'production']),
  port: z.number().min(1).max(65535),
  database: DatabaseConfigSchema,
  api: ApiConfigSchema,
  features: z.object({
    enableLogging: z.boolean(),
    enableMetrics: z.boolean(),
    enableAuth: z.boolean(),
  }),
});

function loadConfig(): AppConfig {
  const config = {
    env: process.env.NODE_ENV || 'development',
    port: parseInt(process.env.PORT || '3000', 10),
    database: {
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432', 10),
      username: process.env.DB_USERNAME || '',
      password: process.env.DB_PASSWORD || '',
      database: process.env.DB_NAME || '',
    },
    api: {
      baseUrl: process.env.API_BASE_URL || 'http://localhost:8080',
      timeout: parseInt(process.env.API_TIMEOUT || '5000', 10),
      retries: parseInt(process.env.API_RETRIES || '3', 10),
    },
    features: {
      enableLogging: process.env.ENABLE_LOGGING === 'true',
      enableMetrics: process.env.ENABLE_METRICS === 'true',
      enableAuth: process.env.ENABLE_AUTH === 'true',
    },
  };

  // Validate configuration
  try {
    return AppConfigSchema.parse(config);
  } catch (error) {
    console.error('Invalid configuration:', error);
    process.exit(1);
  }
}

export const config = loadConfig();
```

## Performance Optimization

### Compilation Performance

```typescript
// tsconfig.json - Performance optimizations
{
  "compilerOptions": {
    // Incremental compilation
    "incremental": true,
    "tsBuildInfoFile": "./.tsbuildinfo",
    
    // Skip type checking for libraries
    "skipLibCheck": true,
    
    // Don't check all files
    "skipDefaultLibCheck": true,
    
    // Faster module resolution
    "moduleResolution": "node",
    
    // Exclude unnecessary files
    "types": ["node"], // Only include specific types
    
    // Disable emit for type-only compilation
    "noEmit": true, // When using with other build tools
  },
  
  // Use project references for large codebases
  "references": [
    { "path": "./packages/shared" },
    { "path": "./packages/client" }
  ],
  
  // Exclude test files in production builds
  "exclude": [
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/__tests__/**",
    "**/node_modules",
    "dist"
  ]
}

// Webpack configuration for faster builds
module.exports = {
  resolve: {
    // Faster module resolution
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
    modules: ['node_modules', 'src'],
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: {
          loader: 'ts-loader',
          options: {
            // Only transpile, skip type checking
            transpileOnly: true,
            // Use project references
            projectReferences: true,
            // Faster compilation
            compilerOptions: {
              module: 'esnext',
            },
          },
        },
        exclude: /node_modules/,
      },
    ],
  },
  // Use fork-ts-checker for type checking in separate process
  plugins: [
    new ForkTsCheckerWebpackPlugin({
      typescript: {
        build: true,
        mode: 'write-references',
      },
    }),
  ],
};
```

### Memory Usage Optimization

```typescript
// Large project configuration
{
  "compilerOptions": {
    // Reduce memory usage
    "preserveWatchOutput": true,
    "assumeChangesOnlyAffectDirectDependencies": true,
    
    // Incremental builds
    "incremental": true,
    "composite": true,
    
    // Skip unnecessary checks
    "skipLibCheck": true,
  },
  
  // Watch options for development
  "watchOptions": {
    "watchFile": "useFsEvents",
    "watchDirectory": "useFsEvents",
    "fallbackPolling": "dynamicPriority",
    "synchronousWatchDirectory": true,
    "excludeDirectories": ["**/node_modules", "_build", "temp"]
  }
}

// Node.js memory options
"scripts": {
  "build": "node --max-old-space-size=4096 ./node_modules/.bin/tsc",
  "build:watch": "node --max-old-space-size=4096 ./node_modules/.bin/tsc --watch"
}
```

Konfigurasi dan tooling yang proper adalah kunci untuk productive TypeScript development. Dengan setup yang tepat, kita bisa mendapatkan maximum benefit dari TypeScript's type system sambil maintaining fast compilation times dan excellent developer experience.