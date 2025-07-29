# [typescript] #07 - TS Modules & Namespaces

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## ES6 Modules
---
TypeScript mendukung ES6 module syntax yang merupakan standar modern untuk organizing code.
### Basic Export/Import

```typescript
// math.ts - Named exports
export function add(x: number, y: number): number {
    return x + y;
}

export function subtract(x: number, y: number): number {
    return x - y;
}

export const PI = 3.14159;

export interface Calculator {
    add(x: number, y: number): number;
    subtract(x: number, y: number): number;
}

// main.ts - Named imports
import { add, subtract, PI, Calculator } from './math';

console.log(add(5, 3));      // 8
console.log(subtract(10, 4)); // 6
console.log(PI);             // 3.14159

// Import dengan alias
import { add as sum, subtract as diff } from './math';

console.log(sum(2, 3));      // 5
console.log(diff(10, 7));    // 3

// Import semua
import * as MathUtils from './math';

console.log(MathUtils.add(1, 2));
console.log(MathUtils.PI);
```

### Default Exports

```typescript
// logger.ts - Default export
export default class Logger {
    private prefix: string;
    
    constructor(prefix: string = '[LOG]') {
        this.prefix = prefix;
    }
    
    log(message: string): void {
        console.log(`${this.prefix} ${message}`);
    }
    
    error(message: string): void {
        console.error(`${this.prefix} ERROR: ${message}`);
    }
}

// utils.ts - Default export dengan function
export default function formatDate(date: Date): string {
    return date.toISOString().split('T')[0];
}

// Mixed exports
export const VERSION = '1.0.0';

// app.ts - Import default exports
import Logger from './logger';           // Default import
import formatDate from './utils';        // Default import
import { VERSION } from './utils';       // Named import

const logger = new Logger('[APP]');
logger.log('Application started');

const today = formatDate(new Date());
logger.log(`Today is ${today}`);
logger.log(`Version: ${VERSION}`);

// Default dengan alias
import DateFormatter from './utils';     // Alias untuk default export
```

### Re-exports

```typescript
// types.ts
export interface User {
    id: number;
    name: string;
    email: string;
}

export interface Product {
    id: number;
    name: string;
    price: number;
}

// services/userService.ts
import { User } from '../types';

export class UserService {
    async getUser(id: number): Promise<User> {
        // Implementation
        return { id, name: 'John', email: 'john@example.com' };
    }
}

// services/productService.ts
import { Product } from '../types';

export class ProductService {
    async getProduct(id: number): Promise<Product> {
        // Implementation
        return { id, name: 'Laptop', price: 1000 };
    }
}

// services/index.ts - Barrel exports
export { UserService } from './userService';
export { ProductService } from './productService';
export * from '../types';  // Re-export all types

// main.ts - Clean imports
import { UserService, ProductService, User, Product } from './services';

const userService = new UserService();
const productService = new ProductService();
```

## Module Resolution
---
TypeScript memiliki berbagai strategi untuk resolving modules.

### Relative vs Non-relative Imports

```typescript
// Relative imports (mulai dengan ./ atau ../)
import { Utils } from './utils';           // Same directory
import { Config } from '../config';        // Parent directory
import { API } from '../../api/client';    // Two levels up

// Non-relative imports (module names)
import * as React from 'react';            // External package
import { Component } from '@angular/core';  // Scoped package
import { helper } from 'my-utils';         // Custom package
```

### Path Mapping (tsconfig.json)

```typescript
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": "./src",
    "paths": {
      "@/*": ["*"],
      "@components/*": ["components/*"],
      "@services/*": ["services/*"],
      "@utils/*": ["utils/*"],
      "@types/*": ["types/*"]
    }
  }
}

// Usage dengan path mapping
import { Button } from '@components/Button';
import { UserService } from '@services/UserService';
import { formatDate } from '@utils/dateUtils';
import { User } from '@types/User';

// Tanpa path mapping (verbose)
import { Button } from '../../../components/Button';
import { UserService } from '../../services/UserService';
```

## Dynamic Imports
---
Dynamic imports memungkinkan loading modules secara asinkron.

```typescript
// Basic dynamic import
async function loadMath() {
    const mathModule = await import('./math');
    return mathModule.add(5, 3);
}

// Conditional loading
async function loadComponent(type: string) {
    let component;
    
    if (type === 'chart') {
        component = await import('./components/Chart');
    } else if (type === 'table') {
        component = await import('./components/Table');
    } else {
        component = await import('./components/Default');
    }
    
    return component.default;
}

// Dynamic import dengan error handling
async function safeImport<T>(modulePath: string): Promise<T | null> {
    try {
        const module = await import(modulePath);
        return module.default || module;
    } catch (error) {
        console.error(`Failed to load module: ${modulePath}`, error);
        return null;
    }
}

// Usage
safeImport<typeof import('./heavyLibrary')>('./heavyLibrary')
    .then(lib => {
        if (lib) {
            lib.performHeavyOperation();
        }
    });

// Code splitting dengan webpack
async function loadUserDashboard() {
    const { UserDashboard } = await import(
        /* webpackChunkName: "user-dashboard" */ './UserDashboard'
    );
    return UserDashboard;
}
```

## Module Augmentation
---
Module augmentation memungkinkan kita menambahkan deklarasi ke existing modules.

```typescript
// Extending existing module
declare module 'lodash' {
    interface LoDashStatic {
        customMethod(value: any): any;
    }
}

// Now lodash has customMethod
import * as _ from 'lodash';
// _.customMethod is now available

// Extending global namespace
declare global {
    interface Window {
        myApp: {
            version: string;
            config: any;
        };
    }
    
    interface Array<T> {
        first(): T | undefined;
        last(): T | undefined;
    }
}

// Implementation
Array.prototype.first = function<T>(this: T[]): T | undefined {
    return this[0];
};

Array.prototype.last = function<T>(this: T[]): T | undefined {
    return this[this.length - 1];
};

// Usage
const numbers = [1, 2, 3, 4, 5];
console.log(numbers.first()); // 1
console.log(numbers.last());  // 5

window.myApp = {
    version: '1.0.0',
    config: {}
};

// Merging with existing module
// original-lib.d.ts
declare module 'original-lib' {
    export function originalFunction(): void;
}

// our-extension.ts
declare module 'original-lib' {
    export function ourExtension(): void;
}

// Now both functions are available
import { originalFunction, ourExtension } from 'original-lib';
```

## Namespaces (Legacy)
---
Meskipun ES6 modules lebih direkomendasikan, namespaces masih berguna dalam beberapa kasus.

```typescript
// geometry.ts
namespace Geometry {
    export interface Point {
        x: number;
        y: number;
    }
    
    export interface Rectangle {
        topLeft: Point;
        bottomRight: Point;
    }
    
    export function distance(p1: Point, p2: Point): number {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    export function area(rect: Rectangle): number {
        const width = rect.bottomRight.x - rect.topLeft.x;
        const height = rect.bottomRight.y - rect.topLeft.y;
        return width * height;
    }
    
    // Nested namespace
    export namespace Utils {
        export function midpoint(p1: Point, p2: Point): Point {
            return {
                x: (p1.x + p2.x) / 2,
                y: (p1.y + p2.y) / 2
            };
        }
    }
}

// Usage
const point1: Geometry.Point = { x: 0, y: 0 };
const point2: Geometry.Point = { x: 3, y: 4 };

console.log(Geometry.distance(point1, point2)); // 5

const rect: Geometry.Rectangle = {
    topLeft: { x: 0, y: 0 },
    bottomRight: { x: 10, y: 5 }
};

console.log(Geometry.area(rect)); // 50

const mid = Geometry.Utils.midpoint(point1, point2);
console.log(mid); // { x: 1.5, y: 2 }

// Alias untuk namespace
import GeomUtils = Geometry.Utils;
const midpoint = GeomUtils.midpoint(point1, point2);
```

## Triple-Slash Directives
---
Triple-slash directives memberikan instruksi kepada compiler.

```typescript
// types.d.ts
/// <reference types="node" />
/// <reference path="./custom-types.d.ts" />

declare namespace MyApp {
    interface Config {
        apiUrl: string;
        debug: boolean;
    }
}

// main.ts
/// <reference path="./types.d.ts" />

const config: MyApp.Config = {
    apiUrl: 'https://api.example.com',
    debug: true
};

// AMD module loading
/// <amd-module name="MyModule" />
export function myFunction() {
    return "Hello from MyModule";
}
```

## Practical Module Patterns
---
### Plugin Architecture

```typescript
// plugin.ts - Plugin interface
export interface Plugin {
    name: string;
    version: string;
    init(context: any): void;
    destroy(): void;
}

// plugins/logger.ts
import { Plugin } from '../plugin';

export default class LoggerPlugin implements Plugin {
    name = 'Logger';
    version = '1.0.0';
    
    init(context: any): void {
        console.log('Logger plugin initialized');
    }
    
    destroy(): void {
        console.log('Logger plugin destroyed');
    }
}

// pluginManager.ts
import { Plugin } from './plugin';

export class PluginManager {
    private plugins: Map<string, Plugin> = new Map();
    
    async loadPlugin(pluginPath: string): Promise<void> {
        try {
            const pluginModule = await import(pluginPath);
            const PluginClass = pluginModule.default;
            const plugin = new PluginClass();
            
            this.plugins.set(plugin.name, plugin);
            plugin.init(this);
            
            console.log(`Plugin ${plugin.name} loaded successfully`);
        } catch (error) {
            console.error(`Failed to load plugin: ${pluginPath}`, error);
        }
    }
    
    unloadPlugin(name: string): void {
        const plugin = this.plugins.get(name);
        if (plugin) {
            plugin.destroy();
            this.plugins.delete(name);
            console.log(`Plugin ${name} unloaded`);
        }
    }
    
    getPlugin(name: string): Plugin | undefined {
        return this.plugins.get(name);
    }
}

// Usage
const manager = new PluginManager();
await manager.loadPlugin('./plugins/logger');
```

### Facade Pattern

```typescript
// services/index.ts - Facade
import { UserService } from './user/UserService';
import { ProductService } from './product/ProductService';
import { OrderService } from './order/OrderService';
import { NotificationService } from './notification/NotificationService';

export class AppServices {
    private static instance: AppServices;
    
    public readonly user: UserService;
    public readonly product: ProductService;
    public readonly order: OrderService;
    public readonly notification: NotificationService;
    
    private constructor() {
        this.user = new UserService();
        this.product = new ProductService();
        this.order = new OrderService();
        this.notification = new NotificationService();
    }
    
    static getInstance(): AppServices {
        if (!AppServices.instance) {
            AppServices.instance = new AppServices();
        }
        return AppServices.instance;
    }
}

// Usage
import { AppServices } from './services';

const services = AppServices.getInstance();
const user = await services.user.getById(1);
const products = await services.product.getAll();
```

### Dependency Injection Container

```typescript
// di-container.ts
type Constructor<T = {}> = new (...args: any[]) => T;
type ServiceKey = string | symbol | Constructor;

export class DIContainer {
    private services = new Map<ServiceKey, any>();
    private factories = new Map<ServiceKey, () => any>();
    
    register<T>(key: ServiceKey, factory: () => T): void {
        this.factories.set(key, factory);
    }
    
    registerSingleton<T>(key: ServiceKey, factory: () => T): void {
        this.register(key, () => {
            if (!this.services.has(key)) {
                this.services.set(key, factory());
            }
            return this.services.get(key);
        });
    }
    
    resolve<T>(key: ServiceKey): T {
        const factory = this.factories.get(key);
        if (!factory) {
            throw new Error(`Service not registered: ${String(key)}`);
        }
        return factory();
    }
}

// Usage
const container = new DIContainer();

// Register services
container.registerSingleton('userService', () => new UserService());
container.registerSingleton('productService', () => new ProductService());

// Resolve services
const userService = container.resolve<UserService>('userService');
const productService = container.resolve<ProductService>('productService');
```

Modules dan namespaces adalah fundamental untuk organizing code TypeScript yang scalable. ES6 modules adalah approach modern yang direkomendasikan, sementara namespaces masih berguna untuk specific use cases dan legacy code.