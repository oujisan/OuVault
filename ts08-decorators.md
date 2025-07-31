# [typescript] #08 - TS Decorators

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## What are Decorators?

Decorators adalah fitur experimental di TypeScript yang memungkinkan kita menambahkan metadata dan memodifikasi classes, methods, properties, dan parameters. Decorator menggunakan syntax `@expression` dan sangat berguna untuk cross-cutting concerns seperti logging, validation, caching, dan dependency injection.

### Enabling Decorators

```typescript
// tsconfig.json
{
  "compilerOptions": {
    "experimentalDecorators": true,
    "emitDecoratorMetadata": true
  }
}
```

## Class Decorators

Class decorators diaplikasikan ke constructor dari class dan bisa modify atau replace class definition.

```typescript
// Basic class decorator
function LogClass(constructor: Function) {
    console.log('Class decorator called');
    console.log(constructor);
}

@LogClass
class MyClass {
    constructor() {
        console.log('MyClass constructor called');
    }
}

// Output:
// Class decorator called
// [Function: MyClass]
// MyClass constructor called (when instantiated)

// Decorator factory (returns decorator)
function Component(name: string) {
    return function(constructor: Function) {
        console.log(`Registering component: ${name}`);
        // Add metadata to class
        (constructor as any).componentName = name;
    };
}

@Component('UserCard')
class UserCard {
    render() {
        return '<div>User Card</div>';
    }
}

console.log((UserCard as any).componentName); // "UserCard"

// Class replacement decorator
function Timestamped<T extends { new(...args: any[]): {} }>(constructor: T) {
    return class extends constructor {
        timestamp = new Date();
        
        constructor(...args: any[]) {
            super(...args);
            console.log(`Instance created at: ${this.timestamp}`);
        }
    };
}

@Timestamped
class User {
    constructor(public name: string) {}
}

const user = new User('John'); // "Instance created at: [timestamp]"
console.log((user as any).timestamp); // Date object
```

## Method Decorators

Method decorators diaplikasikan ke method declarations dan bisa modify method behavior.

```typescript
// Basic method decorator
function Log(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    
    descriptor.value = function(...args: any[]) {
        console.log(`Calling ${propertyName} with args:`, args);
        const result = method.apply(this, args);
        console.log(`${propertyName} returned:`, result);
        return result;
    };
}

class Calculator {
    @Log
    add(x: number, y: number): number {
        return x + y;
    }
    
    @Log
    multiply(x: number, y: number): number {
        return x * y;
    }
}

const calc = new Calculator();
calc.add(2, 3); 
// Output:
// Calling add with args: [2, 3]
// add returned: 5

// Performance measurement decorator
function Measure(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    
    descriptor.value = function(...args: any[]) {
        const start = performance.now();
        const result = method.apply(this, args);
        const end = performance.now();
        console.log(`${propertyName} took ${end - start} milliseconds`);
        return result;
    };
}

class DataProcessor {
    @Measure
    processLargeArray(data: number[]): number[] {
        return data.map(x => x * 2).filter(x => x > 10);
    }
}

// Async method decorator
function AsyncLog(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    
    descriptor.value = async function(...args: any[]) {
        console.log(`Starting async ${propertyName}`);
        try {
            const result = await method.apply(this, args);
            console.log(`${propertyName} completed successfully`);
            return result;
        } catch (error) {
            console.error(`${propertyName} failed:`, error);
            throw error;
        }
    };
}

class ApiService {
    @AsyncLog
    async fetchUser(id: number): Promise<any> {
        const response = await fetch(`/api/users/${id}`);
        return response.json();
    }
}
```

## Property Decorators

Property decorators diaplikasikan ke property declarations.

```typescript
// Basic property decorator
function ReadOnly(target: any, propertyName: string) {
    Object.defineProperty(target, propertyName, {
        writable: false,
        enumerable: true,
        configurable: false
    });
}

// Property validation decorator
function Min(minValue: number) {
    return function(target: any, propertyName: string) {
        let value: number;
        
        const getter = () => value;
        const setter = (newValue: number) => {
            if (newValue < minValue) {
                throw new Error(`${propertyName} must be at least ${minValue}`);
            }
            value = newValue;
        };
        
        Object.defineProperty(target, propertyName, {
            get: getter,
            set: setter,
            enumerable: true,
            configurable: true
        });
    };
}

function Max(maxValue: number) {
    return function(target: any, propertyName: string) {
        let value: number;
        
        const getter = () => value;
        const setter = (newValue: number) => {
            if (newValue > maxValue) {
                throw new Error(`${propertyName} must be at most ${maxValue}`);
            }
            value = newValue;
        };
        
        Object.defineProperty(target, propertyName, {
            get: getter,
            set: setter,
            enumerable: true,
            configurable: true
        });
    };
}

class Product {
    @ReadOnly
    id: number = 1;
    
    @Min(0)
    @Max(1000)
    price: number = 0;
    
    name: string = '';
}

const product = new Product();
// product.id = 2; // Error: Cannot assign to read only property
product.price = 50;  // ✓ Valid
// product.price = -10; // Error: price must be at least 0
// product.price = 1500; // Error: price must be at most 1000

// Observable property decorator
function Observable(target: any, propertyName: string) {
    const privateKey = `_${propertyName}`;
    const listenersKey = `_${propertyName}Listeners`;
    
    target[listenersKey] = [];
    
    Object.defineProperty(target, propertyName, {
        get() {
            return this[privateKey];
        },
        set(value) {
            const oldValue = this[privateKey];
            this[privateKey] = value;
            
            // Notify listeners
            this[listenersKey].forEach((listener: Function) => {
                listener(value, oldValue, propertyName);
            });
        },
        enumerable: true,
        configurable: true
    });
    
    // Add method to subscribe to changes
    target[`on${propertyName.charAt(0).toUpperCase()}${propertyName.slice(1)}Changed`] = 
        function(listener: Function) {
            this[listenersKey].push(listener);
        };
}

class UserModel {
    @Observable
    name: string = '';
    
    @Observable
    email: string = '';
}

const userModel = new UserModel();

// Subscribe to changes
(userModel as any).onNameChanged((newValue: string, oldValue: string) => {
    console.log(`Name changed from "${oldValue}" to "${newValue}"`);
});

userModel.name = 'John'; // "Name changed from "undefined" to "John""
userModel.name = 'Jane'; // "Name changed from "John" to "Jane""
```

## Parameter Decorators

Parameter decorators diaplikasikan ke parameters dalam method atau constructor.

```typescript
// Parameter logging decorator
function LogParam(target: any, propertyName: string, parameterIndex: number) {
    const existingLoggedParams = Reflect.getOwnMetadata('logged_params', target, propertyName) || [];
    existingLoggedParams.push(parameterIndex);
    Reflect.defineMetadata('logged_params', existingLoggedParams, target, propertyName);
}

// Method decorator yang bekerja dengan parameter decorator
function LogParameters(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    const loggedParams = Reflect.getOwnMetadata('logged_params', target, propertyName) || [];
    
    descriptor.value = function(...args: any[]) {
        loggedParams.forEach((paramIndex: number) => {
            console.log(`Parameter ${paramIndex}:`, args[paramIndex]);
        });
        return method.apply(this, args);
    };
}

class UserService {
    @LogParameters
    createUser(@LogParam name: string, @LogParam email: string, age: number) {
        return { name, email, age };
    }
}

const service = new UserService();
service.createUser('John', 'john@example.com', 30);
// Output:
// Parameter 0: John
// Parameter 1: john@example.com

// Validation parameter decorator
function Required(target: any, propertyName: string, parameterIndex: number) {
    const existingRequiredParams = Reflect.getOwnMetadata('required_params', target, propertyName) || [];
    existingRequiredParams.push(parameterIndex);
    Reflect.defineMetadata('required_params', existingRequiredParams, target, propertyName);
}

function ValidateParams(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    const requiredParams = Reflect.getOwnMetadata('required_params', target, propertyName) || [];
    
    descriptor.value = function(...args: any[]) {
        requiredParams.forEach((paramIndex: number) => {
            if (args[paramIndex] === undefined || args[paramIndex] === null) {
                throw new Error(`Parameter at index ${paramIndex} is required`);
            }
        });
        return method.apply(this, args);
    };
}

class OrderService {
    @ValidateParams
    createOrder(@Required customerId: number, @Required items: any[], discount?: number) {
        return { customerId, items, discount };
    }
}

const orderService = new OrderService();
// orderService.createOrder(null, []); // Error: Parameter at index 0 is required
orderService.createOrder(1, []); // ✓ Valid
```

## Decorator Composition

Multiple decorators bisa digunakan pada target yang sama.

```typescript
// Multiple decorators pada class
@Component('UserProfile')
@Timestamped
class UserProfile {
    @Observable
    @Min(0)
    age: number = 0;
    
    @Log
    @Measure
    @AsyncLog
    async updateProfile(@Required userId: number, @Required data: any) {
        // Implementation
        await new Promise(resolve => setTimeout(resolve, 100));
        return { success: true };
    }
}

// Decorator execution order: bottom to top (right to left)
function First() {
    return function(target: any, propertyName: string, descriptor: PropertyDescriptor) {
        console.log('First decorator');
    };
}

function Second() {
    return function(target: any, propertyName: string, descriptor: PropertyDescriptor) {
        console.log('Second decorator');
    };
}

class Example {
    @First()  // Executed second
    @Second() // Executed first
    method() {}
}
// Output:
// Second decorator
// First decorator
```

## Practical Decorator Examples

### Caching Decorator

```typescript
function Cache(ttl: number = 60000) { // TTL in milliseconds
    return function(target: any, propertyName: string, descriptor: PropertyDescriptor) {
        const method = descriptor.value;
        const cache = new Map<string, { value: any; timestamp: number }>();
        
        descriptor.value = function(...args: any[]) {
            const key = JSON.stringify(args);
            const cached = cache.get(key);
            
            if (cached && (Date.now() - cached.timestamp) < ttl) {
                console.log(`Cache hit for ${propertyName}`);
                return cached.value;
            }
            
            console.log(`Cache miss for ${propertyName}`);
            const result = method.apply(this, args);
            cache.set(key, { value: result, timestamp: Date.now() });
            
            return result;
        };
    };
}

class DatabaseService {
    @Cache(30000) // Cache for 30 seconds
    async getUser(id: number): Promise<any> {
        console.log(`Fetching user ${id} from database`);
        // Simulate database call
        await new Promise(resolve => setTimeout(resolve, 1000));
        return { id, name: `User ${id}` };
    }
}
```

### Retry Decorator

```typescript
function Retry(maxAttempts: number = 3, delay: number = 1000) {
    return function(target: any, propertyName: string, descriptor: PropertyDescriptor) {
        const method = descriptor.value;
        
        descriptor.value = async function(...args: any[]) {
            let lastError: any;
            
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    return await method.apply(this, args);
                } catch (error) {
                    lastError = error;
                    console.log(`${propertyName} attempt ${attempt} failed:`, error.message);
                    
                    if (attempt < maxAttempts) {
                        console.log(`Retrying in ${delay}ms...`);
                        await new Promise(resolve => setTimeout(resolve, delay));
                    }
                }
            }
            
            throw new Error(`${propertyName} failed after ${maxAttempts} attempts: ${lastError.message}`);
        };
    };
}

class NetworkService {
    @Retry(3, 2000)
    async fetchData(url: string): Promise<any> {
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    }
}
```

### Authorization Decorator

```typescript
function RequireRole(role: string) {
    return function(target: any, propertyName: string, descriptor: PropertyDescriptor) {
        const method = descriptor.value;
        
        descriptor.value = function(...args: any[]) {
            // Get current user context (this would come from your auth system)
            const currentUser = getCurrentUser(); // Assumed function
            
            if (!currentUser) {
                throw new Error('User not authenticated');
            }
            
            if (!currentUser.roles.includes(role)) {
                throw new Error(`Access denied. Required role: ${role}`);
            }
            
            return method.apply(this, args);
        };
    };
}

function getCurrentUser() {
    // Mock implementation
    return {
        id: 1,
        name: 'John',
        roles: ['user', 'admin']
    };
}

class AdminService {
    @RequireRole('admin')
    deleteUser(userId: number): boolean {
        console.log(`Deleting user ${userId}`);
        return true;
    }
    
    @RequireRole('moderator')
    moderateContent(contentId: number): void {
        console.log(`Moderating content ${contentId}`);
    }
}
```

### Model Validation Decorator

```typescript
// Validation metadata storage
const validationMetadata = new Map<any, Map<string, any[]>>();

function getValidationMetadata(target: any): Map<string, any[]> {
    if (!validationMetadata.has(target.constructor)) {
        validationMetadata.set(target.constructor, new Map());
    }
    return validationMetadata.get(target.constructor)!;
}

// Validation decorators
function IsEmail(target: any, propertyName: string) {
    const metadata = getValidationMetadata(target);
    if (!metadata.has(propertyName)) {
        metadata.set(propertyName, []);
    }
    metadata.get(propertyName)!.push({
        type: 'email',
        validator: (value: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
        message: `${propertyName} must be a valid email address`
    });
}

function MinLength(length: number) {
    return function(target: any, propertyName: string) {
        const metadata = getValidationMetadata(target);
        if (!metadata.has(propertyName)) {
            metadata.set(propertyName, []);
        }
        metadata.get(propertyName)!.push({
            type: 'minLength',
            validator: (value: string) => value && value.length >= length,
            message: `${propertyName} must be at least ${length} characters long`
        });
    };
}

function IsNumber(target: any, propertyName: string) {
    const metadata = getValidationMetadata(target);
    if (!metadata.has(propertyName)) {
        metadata.set(propertyName, []);
    }
    metadata.get(propertyName)!.push({
        type: 'number',
        validator: (value: any) => !isNaN(Number(value)),
        message: `${propertyName} must be a valid number`
    });
}

// Validation function
function validate(obj: any): { isValid: boolean; errors: string[] } {
    const metadata = getValidationMetadata(obj);
    const errors: string[] = [];
    
    for (const [propertyName, validators] of metadata.entries()) {
        const value = obj[propertyName];
        
        for (const validator of validators) {
            if (!validator.validator(value)) {
                errors.push(validator.message);
            }
        }
    }
    
    return {
        isValid: errors.length === 0,
        errors
    };
}

// Usage
class UserRegistrationModel {
    @IsEmail
    email: string = '';
    
    @MinLength(8)
    password: string = '';
    
    @MinLength(2)
    firstName: string = '';
    
    @IsNumber
    age: string = '';
}

const user = new UserRegistrationModel();
user.email = 'invalid-email';
user.password = '123';
user.firstName = 'J';
user.age = 'not-a-number';

const result = validate(user);
console.log(result);
// {
//   isValid: false,
//   errors: [
//     'email must be a valid email address',
//     'password must be at least 8 characters long',
//     'firstName must be at least 2 characters long',
//     'age must be a valid number'
//   ]
// }
```

## Reflect Metadata API

Untuk decorators yang lebih advanced, kita sering menggunakan reflect-metadata library.

```typescript
// Install: npm install reflect-metadata
import 'reflect-metadata';

// Metadata decorator
function SetMetadata(key: string, value: any) {
    return function(target: any, propertyName?: string) {
        if (propertyName) {
            Reflect.defineMetadata(key, value, target, propertyName);
        } else {
            Reflect.defineMetadata(key, value, target);
        }
    };
}

function GetMetadata(key: string) {
    return function(target: any, propertyName?: string) {
        if (propertyName) {
            return Reflect.getMetadata(key, target, propertyName);
        } else {
            return Reflect.getMetadata(key, target);
        }
    };
}

// Route decorator untuk web framework
function Route(path: string, method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'GET') {
    return function(target: any, propertyName: string, descriptor: PropertyDescriptor) {
        Reflect.defineMetadata('route:path', path, target, propertyName);
        Reflect.defineMetadata('route:method', method, target, propertyName);
    };
}

function Controller(basePath: string) {
    return function(constructor: Function) {
        Reflect.defineMetadata('controller:basePath', basePath, constructor);
    };
}

// Usage
@Controller('/api/users')
class UserController {
    @Route('/', 'GET')
    getAllUsers() {
        return 'Get all users';
    }
    
    @Route('/:id', 'GET')
    getUserById() {
        return 'Get user by ID';
    }
    
    @Route('/', 'POST')
    createUser() {
        return 'Create user';
    }
}

// Router setup function
function setupRoutes(controllerClass: any) {
    const basePath = Reflect.getMetadata('controller:basePath', controllerClass) || '';
    const instance = new controllerClass();
    
    const methodNames = Object.getOwnPropertyNames(controllerClass.prototype)
        .filter(name => name !== 'constructor');
    
    methodNames.forEach(methodName => {
        const routePath = Reflect.getMetadata('route:path', controllerClass.prototype, methodName);
        const routeMethod = Reflect.getMetadata('route:method', controllerClass.prototype, methodName);
        
        if (routePath && routeMethod) {
            const fullPath = basePath + routePath;
            console.log(`${routeMethod} ${fullPath} -> ${controllerClass.name}.${methodName}`);
            
            // Here you would register the route with your web framework
            // app[routeMethod.toLowerCase()](fullPath, instance[methodName].bind(instance));
        }
    });
}

setupRoutes(UserController);
// Output:
// GET /api/users/ -> UserController.getAllUsers
// GET /api/users/:id -> UserController.getUserById
// POST /api/users/ -> UserController.createUser
```

## Dependency Injection with Decorators

```typescript
import 'reflect-metadata';

// Service registry
const serviceRegistry = new Map<any, any>();

// Injectable decorator
function Injectable(target: any) {
    Reflect.defineMetadata('injectable', true, target);
    return target;
}

// Inject decorator
function Inject(token?: any) {
    return function(target: any, propertyKey: string | symbol | undefined, parameterIndex: number) {
        const existingTokens = Reflect.getOwnMetadata('inject-tokens', target) || [];
        existingTokens[parameterIndex] = token || Reflect.getMetadata('design:paramtypes', target)[parameterIndex];
        Reflect.defineMetadata('inject-tokens', existingTokens, target);
    };
}

// Container class
class DIContainer {
    private instances = new Map<any, any>();
    
    register<T>(token: any, implementation: new (...args: any[]) => T): void {
        serviceRegistry.set(token, implementation);
    }
    
    resolve<T>(token: any): T {
        if (this.instances.has(token)) {
            return this.instances.get(token);
        }
        
        const implementation = serviceRegistry.get(token);
        if (!implementation) {
            throw new Error(`No implementation found for ${token.name || token}`);
        }
        
        const isInjectable = Reflect.getMetadata('injectable', implementation);
        if (!isInjectable) {
            throw new Error(`${implementation.name} is not injectable`);
        }
        
        const tokens = Reflect.getMetadata('inject-tokens', implementation) || [];
        const paramTypes = Reflect.getMetadata('design:paramtypes', implementation) || [];
        
        const dependencies = paramTypes.map((type: any, index: number) => {
            const token = tokens[index] || type;
            return this.resolve(token);
        });
        
        const instance = new implementation(...dependencies);
        this.instances.set(token, instance);
        
        return instance;
    }
}

// Usage
abstract class Logger {
    abstract log(message: string): void;
}

@Injectable
class ConsoleLogger implements Logger {
    log(message: string): void {
        console.log(`[LOG] ${message}`);
    }
}

abstract class Database {
    abstract query(sql: string): any[];
}

@Injectable
class PostgresDatabase implements Database {
    query(sql: string): any[] {
        console.log(`Executing query: ${sql}`);
        return [];
    }
}

@Injectable
class UserService {
    constructor(
        @Inject(Logger) private logger: Logger,
        @Inject(Database) private database: Database
    ) {}
    
    getUsers(): any[] {
        this.logger.log('Getting all users');
        return this.database.query('SELECT * FROM users');
    }
    
    createUser(userData: any): void {
        this.logger.log('Creating new user');
        this.database.query('INSERT INTO users ...');
    }
}

// Setup container
const container = new DIContainer();
container.register(Logger, ConsoleLogger);
container.register(Database, PostgresDatabase);
container.register(UserService, UserService);

// Resolve and use
const userService = container.resolve<UserService>(UserService);
userService.getUsers();
// Output:
// [LOG] Getting all users
// Executing query: SELECT * FROM users
```

## Best Practices

```typescript
// 1. Keep decorators simple and focused
function SimpleLog(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    descriptor.value = function(...args: any[]) {
        console.log(`Calling ${propertyName}`);
        return method.apply(this, args);
    };
}

// 2. Use decorator factories for configuration
function ConfigurableCache(options: { ttl?: number; maxSize?: number } = {}) {
    const { ttl = 60000, maxSize = 100 } = options;
    
    return function(target: any, propertyName: string, descriptor: PropertyDescriptor) {
        // Implementation with configurable options
    };
}

// 3. Provide type safety
function TypedDecorator<T extends (...args: any[]) => any>(
    target: any,
    propertyName: string,
    descriptor: TypedPropertyDescriptor<T>
): void {
    // Type-safe decorator implementation
}

// 4. Handle errors gracefully
function SafeDecorator(target: any, propertyName: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    
    descriptor.value = function(...args: any[]) {
        try {
            return method.apply(this, args);
        } catch (error) {
            console.error(`Error in ${propertyName}:`, error);
            // Handle error appropriately
            throw error;
        }
    };
}

// 5. Use composition for complex functionality
function CombinedDecorator(...decorators: any[]) {
    return function(target: any, propertyName: string, descriptor: PropertyDescriptor) {
        decorators.reverse().forEach(decorator => {
            decorator(target, propertyName, descriptor);
        });
    };
}

class Example {
    @CombinedDecorator(Log, Measure, Cache())
    complexMethod() {
        // This method will be logged, measured, and cached
    }
}
```

Decorators adalah fitur yang sangat powerful untuk implementing cross-cutting concerns dan metadata-driven programming. Meskipun masih experimental, mereka sangat berguna untuk framework development, dependency injection, validation, dan banyak pattern lainnya yang membutuhkan clean separation of concerns.