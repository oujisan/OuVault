# [typescript] #10 - TS Best Practices & Patterns

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## Type Safety Best Practices
---
### Prefer Type Assertions Over Any

```typescript
// ❌ Bad - using any
function processData(data: any) {
    return data.someProperty;
}

// ✅ Good - using proper types
interface ApiResponse {
    data: unknown;
    status: number;
}

function processApiResponse(response: ApiResponse) {
    // Type narrowing
    if (typeof response.data === 'object' && response.data !== null) {
        return response.data;
    }
    throw new Error('Invalid response data');
}

// ✅ Good - using type assertions when necessary
function processJsonResponse(jsonString: string) {
    const parsed: unknown = JSON.parse(jsonString);
    
    // Assert with validation
    if (isUserData(parsed)) {
        return parsed; // Now typed as UserData
    }
    
    throw new Error('Invalid user data');
}

function isUserData(data: unknown): data is UserData {
    return (
        typeof data === 'object' &&
        data !== null &&
        'id' in data &&
        'name' in data
    );
}
```

### Use Strict Configuration

```typescript
// tsconfig.json - Always use strict mode
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noImplicitReturns": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "exactOptionalPropertyTypes": true
  }
}

// ❌ Bad - implicit any
function calculateTotal(items) {
    return items.reduce((sum, item) => sum + item.price, 0);
}

// ✅ Good - explicit types
interface Item {
    price: number;
    name: string;
}

function calculateTotal(items: Item[]): number {
    return items.reduce((sum, item) => sum + item.price, 0);
}

// ❌ Bad - not handling null/undefined
function getFullName(user: User) {
    return user.firstName + ' ' + user.lastName; // Could throw if user is null
}

// ✅ Good - proper null checking
function getFullName(user: User | null): string {
    if (!user) {
        return 'Anonymous';
    }
    return `${user.firstName} ${user.lastName}`;
}
```

### Prefer Unknown Over Any

```typescript
// ❌ Bad - using any loses all type safety
function processUserInput(input: any) {
    console.log(input.toUpperCase()); // Runtime error if input is not string
    return input;
}

// ✅ Good - using unknown requires type checking
function processUserInput(input: unknown): string {
    if (typeof input === 'string') {
        return input.toUpperCase(); // Safe because we checked the type
    }
    
    if (typeof input === 'number') {
        return input.toString();
    }
    
    throw new Error('Invalid input type');
}

// ✅ Better - using generic with constraints
function processValue<T extends string | number>(value: T): string {
    if (typeof value === 'string') {
        return value.toUpperCase();
    }
    return value.toString();
}
```

## Interface and Type Design
---
### Composition Over Inheritance

```typescript
// ❌ Bad - deep inheritance hierarchy
class Animal {
    name: string;
    constructor(name: string) { this.name = name; }
}

class Mammal extends Animal {
    warmBlooded: boolean = true;
}

class Dog extends Mammal {
    breed: string;
    constructor(name: string, breed: string) {
        super(name);
        this.breed = breed;
    }
}

// ✅ Good - composition with interfaces
interface Named {
    name: string;
}

interface WarmBlooded {
    warmBlooded: boolean;
}

interface Breed {
    breed: string;
}

// Compose interfaces
interface Dog extends Named, WarmBlooded, Breed {}

// Implementation
class DogImpl implements Dog {
    constructor(
        public name: string,
        public breed: string,
        public warmBlooded: boolean = true
    ) {}
}

// Even better - using type composition
type Animal = Named & {
    species: string;
};

type Mammal = Animal & WarmBlooded;
type Dog = Mammal & Breed;
```

### Prefer Immutable Data Structures

```typescript
// ❌ Bad - mutable operations
interface User {
    id: number;
    name: string;
    email: string;
    preferences: { theme: string; notifications: boolean };
}

function updateUserEmail(user: User, newEmail: string): void {
    user.email = newEmail; // Mutates original object
}

// ✅ Good - immutable updates
interface ReadonlyUser {
    readonly id: number;
    readonly name: string;
    readonly email: string;
    readonly preferences: Readonly<{
        theme: string;
        notifications: boolean;
    }>;
}

function updateUserEmail(user: ReadonlyUser, newEmail: string): ReadonlyUser {
    return {
        ...user,
        email: newEmail // Returns new object
    };
}

// ✅ Better - using utility types
type UpdateUser<T> = {
    readonly [K in keyof T]: T[K];
};

function updateUser<K extends keyof User>(
    user: UpdateUser<User>,
    key: K,
    value: User[K]
): UpdateUser<User> {
    return { ...user, [key]: value };
}

// Usage
const updatedUser = updateUser(user, 'email', 'new@email.com');
```

### Use Discriminated Unions for State Management

```typescript
// ❌ Bad - using optional properties for state
interface LoadingState {
    isLoading?: boolean;
    data?: User[];
    error?: string;
}

// Problems: all states are optional, invalid combinations possible

// ✅ Good - discriminated unions
type AsyncState<T> =
    | { status: 'idle' }
    | { status: 'loading' }
    | { status: 'success'; data: T }
    | { status: 'error'; error: string };

function handleUserState(state: AsyncState<User[]>) {
    switch (state.status) {
        case 'idle':
            return 'No data loaded yet';
        
        case 'loading':
            return 'Loading users...';
        
        case 'success':
            return `Loaded ${state.data.length} users`; // data is guaranteed to exist
        
        case 'error':
            return `Error: ${state.error}`; // error is guaranteed to exist
        
        default:
            // Exhaustiveness check
            const _exhaustive: never = state;
            return _exhaustive;
    }
}
```

## Error Handling Patterns
---
### Result Pattern for Error Handling

```typescript
// Define Result type
type Result<T, E = Error> = 
    | { success: true; data: T }
    | { success: false; error: E };

// Helper functions
function Ok<T>(data: T): Result<T, never> {
    return { success: true, data };
}

function Err<E>(error: E): Result<never, E> {
    return { success: false, error };
}

// Usage in functions
async function fetchUser(id: number): Promise<Result<User, string>> {
    try {
        const response = await fetch(`/api/users/${id}`);
        
        if (!response.ok) {
            return Err(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const user = await response.json();
        return Ok(user);
    } catch (error) {
        return Err(`Network error: ${error.message}`);
    }
}

// Safe usage without throwing
async function handleUserFetch(id: number) {
    const result = await fetchUser(id);
    
    if (result.success) {
        console.log('User:', result.data.name); // Type-safe access
    } else {
        console.error('Failed to fetch user:', result.error);
    }
}

// Chainable operations
function mapResult<T, U, E>(
    result: Result<T, E>,
    fn: (data: T) => U
): Result<U, E> {
    if (result.success) {
        return Ok(fn(result.data));
    }
    return result;
}

function flatMapResult<T, U, E>(
    result: Result<T, E>,
    fn: (data: T) => Result<U, E>
): Result<U, E> {
    if (result.success) {
        return fn(result.data);
    }
    return result;
}

// Usage
const result = await fetchUser(1);
const processedResult = mapResult(result, user => user.name.toUpperCase());
```

### Option Pattern for Nullable Values

```typescript
// Option type
type Option<T> = 
    | { some: true; value: T }
    | { some: false };

// Helper functions
function Some<T>(value: T): Option<T> {
    return { some: true, value };
}

function None<T>(): Option<T> {
    return { some: false };
}

// Safe array access
function getArrayItem<T>(array: T[], index: number): Option<T> {
    if (index >= 0 && index < array.length) {
        return Some(array[index]);
    }
    return None();
}

// Usage
const users = ['Alice', 'Bob', 'Charlie'];
const firstUser = getArrayItem(users, 0);

if (firstUser.some) {
    console.log('First user:', firstUser.value); // Type-safe
} else {
    console.log('No user found');
}

// Option utilities
function mapOption<T, U>(option: Option<T>, fn: (value: T) => U): Option<U> {
    if (option.some) {
        return Some(fn(option.value));
    }
    return None();
}

function filterOption<T>(option: Option<T>, predicate: (value: T) => boolean): Option<T> {
    if (option.some && predicate(option.value)) {
        return option;
    }
    return None();
}

function getOrElse<T>(option: Option<T>, defaultValue: T): T {
    return option.some ? option.value : defaultValue;
}
```

## Performance Patterns
---
### Lazy Loading with Generics

```typescript
// Lazy wrapper
class Lazy<T> {
    private _value?: T;
    private _computed = false;
    
    constructor(private factory: () => T) {}
    
    get value(): T {
        if (!this._computed) {
            this._value = this.factory();
            this._computed = true;
        }
        return this._value!;
    }
    
    get isComputed(): boolean {
        return this._computed;
    }
    
    reset(): void {
        this._computed = false;
        this._value = undefined;
    }
}

// Usage
class ExpensiveService {
    private _expensiveData = new Lazy(() => {
        console.log('Computing expensive data...');
        return Array.from({ length: 1000000 }, (_, i) => i * i);
    });
    
    get expensiveData(): number[] {
        return this._expensiveData.value;
    }
    
    resetCache(): void {
        this._expensiveData.reset();
    }
}

const service = new ExpensiveService();
// Data is not computed yet
console.log('Service created');

// Data is computed only when accessed
const data = service.expensiveData;
```

### Memoization Pattern

```typescript
// Generic memoization decorator
function memoize<Args extends unknown[], Return>(
    fn: (...args: Args) => Return,
    getKey?: (...args: Args) => string
): (...args: Args) => Return {
    const cache = new Map<string, Return>();
    
    return (...args: Args): Return => {
        const key = getKey ? getKey(...args) : JSON.stringify(args);
        
        if (cache.has(key)) {
            return cache.get(key)!;
        }
        
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
}

// Usage
const expensiveCalculation = memoize((x: number, y: number): number => {
    console.log(`Computing ${x} + ${y}`);
    // Simulate expensive operation
    for (let i = 0; i < 1000000; i++) {
        Math.sqrt(i);
    }
    return x + y;
});

console.log(expensiveCalculation(5, 3)); // Computed
console.log(expensiveCalculation(5, 3)); // Cached
console.log(expensiveCalculation(2, 4)); // Computed
console.log(expensiveCalculation(2, 4)); // Cached

// Async memoization
function memoizeAsync<Args extends unknown[], Return>(
    fn: (...args: Args) => Promise<Return>,
    getKey?: (...args: Args) => string
): (...args: Args) => Promise<Return> {
    const cache = new Map<string, Promise<Return>>();
    
    return (...args: Args): Promise<Return> => {
        const key = getKey ? getKey(...args) : JSON.stringify(args);
        
        if (cache.has(key)) {
            return cache.get(key)!;
        }
        
        const promise = fn(...args);
        cache.set(key, promise);
        
        // Remove from cache if promise rejects
        promise.catch(() => cache.delete(key));
        
        return promise;
    };
}
```

## Design Patterns
---
### Builder Pattern

```typescript
// Builder pattern dengan fluent interface
class QueryBuilder<T> {
    private selectFields: (keyof T)[] = [];
    private whereConditions: string[] = [];
    private orderByFields: { field: keyof T; direction: 'ASC' | 'DESC' }[] = [];
    private limitCount?: number;
    
    select<K extends keyof T>(...fields: K[]): QueryBuilder<Pick<T, K>> {
        this.selectFields = fields;
        return this as any;
    }
    
    where(condition: string): QueryBuilder<T> {
        this.whereConditions.push(condition);
        return this;
    }
    
    orderBy(field: keyof T, direction: 'ASC' | 'DESC' = 'ASC'): QueryBuilder<T> {
        this.orderByFields.push({ field, direction });
        return this;
    }
    
    limit(count: number): QueryBuilder<T> {
        this.limitCount = count;
        return this;
    }
    
    build(): string {
        let query = 'SELECT ';
        query += this.selectFields.length > 0 
            ? this.selectFields.join(', ') 
            : '*';
        
        query += ' FROM table';
        
        if (this.whereConditions.length > 0) {
            query += ` WHERE ${this.whereConditions.join(' AND ')}`;
        }
        
        if (this.orderByFields.length > 0) {
            const orderBy = this.orderByFields
                .map(({ field, direction }) => `${String(field)} ${direction}
```