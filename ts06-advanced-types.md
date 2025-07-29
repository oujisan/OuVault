# [typescript] #06 - TS Advanced Types

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## Union Types

Union types memungkinkan variable menerima lebih dari satu tipe.

```typescript
// Basic union types
type ID = string | number;
type Status = "loading" | "success" | "error";

function printID(id: ID): void {
    console.log(`ID: ${id}`);
}

printID("USER123");  // ✓ Valid
printID(12345);      // ✓ Valid
// printID(true);    // ✗ Error

// Union dengan objects
type Cat = {
    type: "cat";
    meow: () => void;
};

type Dog = {
    type: "dog";
    bark: () => void;
};

type Pet = Cat | Dog;

function handlePet(pet: Pet): void {
    // Type narrowing dengan discriminated unions
    if (pet.type === "cat") {
        pet.meow(); // TypeScript tahu ini Cat
    } else {
        pet.bark(); // TypeScript tahu ini Dog
    }
}

// Array dengan union types
type StringOrNumber = string | number;
const mixedArray: StringOrNumber[] = ["hello", 42, "world", 123];

// Function overloading dengan unions
function format(value: string): string;
function format(value: number): string;
function format(value: string | number): string {
    if (typeof value === "string") {
        return value.toUpperCase();
    } else {
        return value.toFixed(2);
    }
}
```

## Intersection Types

Intersection types menggabungkan multiple types menjadi satu.

```typescript
// Basic intersection
type Person = {
    name: string;
    age: number;
};

type Employee = {
    employeeId: string;
    department: string;
};

type Staff = Person & Employee;

const staff: Staff = {
    name: "John Doe",
    age: 30,
    employeeId: "EMP001",
    department: "Engineering"
};

// Intersection dengan methods
type Flyable = {
    fly(): void;
    altitude: number;
};

type Swimmable = {
    swim(): void;
    depth: number;
};

type Amphibious = Flyable & Swimmable;

class Duck implements Amphibious {
    altitude: number = 0;
    depth: number = 0;
    
    fly(): void {
        this.altitude = 100;
        console.log("Flying!");
    }
    
    swim(): void {
        this.depth = 5;
        console.log("Swimming!");
    }
}

// Intersection dengan generics
type WithTimestamp<T> = T & {
    createdAt: Date;
    updatedAt: Date;
};

type User = {
    id: number;
    name: string;
    email: string;
};

type UserWithTimestamp = WithTimestamp<User>;

const user: UserWithTimestamp = {
    id: 1,
    name: "Alice",
    email: "alice@example.com",
    createdAt: new Date(),
    updatedAt: new Date()
};
```

## Type Guards

Type guards membantu TypeScript memahami tipe yang lebih spesifik dalam runtime.

```typescript
// typeof type guards
function processValue(value: string | number): string {
    if (typeof value === "string") {
        // Di dalam block ini, TypeScript tahu value adalah string
        return value.toUpperCase();
    } else {
        // Di sini TypeScript tahu value adalah number
        return value.toFixed(2);
    }
}

// instanceof type guards
class Car {
    drive(): void {
        console.log("Driving a car");
    }
}

class Plane {
    fly(): void {
        console.log("Flying a plane");
    }
}

function operateVehicle(vehicle: Car | Plane): void {
    if (vehicle instanceof Car) {
        vehicle.drive(); // TypeScript tahu ini Car
    } else {
        vehicle.fly();   // TypeScript tahu ini Plane
    }
}

// in operator type guards
type Fish = {
    swim: () => void;
    fins: number;
};

type Bird = {
    fly: () => void;
    wings: number;
};

function move(animal: Fish | Bird): void {
    if ("swim" in animal) {
        animal.swim(); // TypeScript tahu ini Fish
    } else {
        animal.fly();  // TypeScript tahu ini Bird
    }
}

// Custom type guards
interface User {
    id: number;
    name: string;
    email: string;
}

interface Admin {
    id: number;
    name: string;
    email: string;
    permissions: string[];
}

// Type predicate function
function isAdmin(user: User | Admin): user is Admin {
    return 'permissions' in user;
}

function handleUser(user: User | Admin): void {
    if (isAdmin(user)) {
        // TypeScript tahu user adalah Admin
        console.log(`Admin ${user.name} has ${user.permissions.length} permissions`);
    } else {
        // TypeScript tahu user adalah User
        console.log(`Regular user: ${user.name}`);
    }
}
```

## Mapped Types

Mapped types memungkinkan kita membuat tipe baru berdasarkan tipe yang sudah ada.

```typescript
// Basic mapped type
type Optional<T> = {
    [K in keyof T]?: T[K];
};

type User = {
    id: number;
    name: string;
    email: string;
    age: number;
};

type OptionalUser = Optional<User>;
// Result: { id?: number; name?: string; email?: string; age?: number; }

// Readonly mapped type
type Readonly<T> = {
    readonly [K in keyof T]: T[K];
};

type ReadonlyUser = Readonly<User>;
// Result: { readonly id: number; readonly name: string; ... }

// Pick specific properties
type Pick<T, K extends keyof T> = {
    [P in K]: T[P];
};

type UserSummary = Pick<User, "id" | "name">;
// Result: { id: number; name: string; }

// Omit specific properties
type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

type UserWithoutId = Omit<User, "id">;
// Result: { name: string; email: string; age: number; }

// Custom mapped types
type Nullable<T> = {
    [K in keyof T]: T[K] | null;
};

type StringKeys<T> = {
    [K in keyof T]: string;
};

type NullableUser = Nullable<User>;
// Result: { id: number | null; name: string | null; ... }

type UserStringKeys = StringKeys<User>;
// Result: { id: string; name: string; email: string; age: string; }
```

## Conditional Types

Conditional types memilih tipe berdasarkan kondisi.

```typescript
// Basic conditional type
type IsArray<T> = T extends any[] ? true : false;

type Test1 = IsArray<string>;    // false
type Test2 = IsArray<number[]>;  // true
type Test3 = IsArray<string[]>;  // true

// Extract array element type
type ArrayElement<T> = T extends (infer U)[] ? U : never;

type StringElement = ArrayElement<string[]>;  // string
type NumberElement = ArrayElement<number[]>;  // number
type NotArray = ArrayElement<string>;         // never

// Function return type extraction
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

function getString(): string { return "hello"; }
function getNumber(): number { return 42; }

type StringReturn = ReturnType<typeof getString>; // string
type NumberReturn = ReturnType<typeof getNumber>; // number

// Distributive conditional types
type ToArray<T> = T extends any ? T[] : never;

type StringOrNumberArray = ToArray<string | number>;
// Result: string[] | number[] (distributed over union)

// Non-null type
type NonNull<T> = T extends null | undefined ? never : T;

type NotNull1 = NonNull<string | null>;      // string
type NotNull2 = NonNull<number | undefined>; // number
type NotNull3 = NonNull<boolean | null | undefined>; // boolean

// Advanced conditional with multiple conditions
type DeepReadonly<T> = {
    readonly [K in keyof T]: T[K] extends object 
        ? DeepReadonly<T[K]> 
        : T[K];
};

interface NestedObject {
    user: {
        profile: {
            name: string;
            settings: {
                theme: string;
            };
        };
    };
}

type DeepReadonlyNested = DeepReadonly<NestedObject>;
// All properties at all levels become readonly
```

## Template Literal Types

Template literal types memungkinkan kita membuat tipe string yang sophisticated.

```typescript
// Basic template literal types
type Greeting = `Hello, ${string}!`;

const greeting1: Greeting = "Hello, World!";     // ✓ Valid
const greeting2: Greeting = "Hello, TypeScript!"; // ✓ Valid
// const greeting3: Greeting = "Hi, World!";      // ✗ Error

// With literal unions
type Color = "red" | "green" | "blue";
type Size = "small" | "medium" | "large";

type ColoredSize = `${Color}-${Size}`;
// Result: "red-small" | "red-medium" | "red-large" | 
//         "green-small" | "green-medium" | "green-large" |
//         "blue-small" | "blue-medium" | "blue-large"

// Event names
type EventName<T extends string> = `on${Capitalize<T>}`;

type ClickEvent = EventName<"click">;    // "onClick"
type HoverEvent = EventName<"hover">;    // "onHover"

// CSS properties
type CSSProperties = "margin" | "padding" | "border";
type CSSDirections = "top" | "right" | "bottom" | "left";

type CSSPropertyWithDirection = `${CSSProperties}-${CSSDirections}`;
// Result: "margin-top" | "margin-right" | ... | "border-left"

// API endpoints
type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE";
type APIEndpoint<T extends string> = `/${T}`;

type UserEndpoints = APIEndpoint<"users" | "users/profile" | "users/settings">;
// Result: "/users" | "/users/profile" | "/users/settings"

// Template literal dengan utility types
type SnakeToCamelCase<S extends string> = S extends `${infer T}_${infer U}`
    ? `${T}${Capitalize<SnakeToCamelCase<U>>}`
    : S;

type CamelCase1 = SnakeToCamelCase<"user_name">;        // "userName"
type CamelCase2 = SnakeToCamelCase<"api_key_secret">;   // "apiKeySecret"
```

## Utility Types

TypeScript menyediakan banyak utility types built-in untuk transformasi tipe.

```typescript
interface User {
    id: number;
    name: string;
    email: string;
    age: number;
    isActive: boolean;
}

// Partial - membuat semua properties optional
type PartialUser = Partial<User>;
// { id?: number; name?: string; email?: string; age?: number; isActive?: boolean; }

function updateUser(id: number, updates: Partial<User>): void {
    // Implementation here
}

updateUser(1, { name: "John" }); // ✓ Valid
updateUser(2, { age: 30, isActive: false }); // ✓ Valid

// Required - membuat semua properties required
type RequiredUser = Required<PartialUser>;
// Back to: { id: number; name: string; email: string; age: number; isActive: boolean; }

// Pick - memilih properties tertentu
type UserProfile = Pick<User, "name" | "email">;
// { name: string; email: string; }

// Omit - menghilangkan properties tertentu
type CreateUserRequest = Omit<User, "id">;
// { name: string; email: string; age: number; isActive: boolean; }

// Record - membuat object type dengan keys dan values tertentu
type UserRoles = Record<string, string[]>;
// { [x: string]: string[]; }

const roles: UserRoles = {
    admin: ["read", "write", "delete"],
    user: ["read"],
    moderator: ["read", "write"]
};

type StatusMap = Record<"loading" | "success" | "error", string>;
// { loading: string; success: string; error: string; }

// Exclude - menghilangkan tipe dari union
type PrimaryColors = "red" | "green" | "blue";
type SecondaryColors = "orange" | "purple" | "green";

type PureSecondary = Exclude<SecondaryColors, PrimaryColors>;
// "orange" | "purple"

// Extract - mengambil tipe yang ada di kedua union
type CommonColors = Extract<PrimaryColors, SecondaryColors>;
// "green"

// NonNullable - menghilangkan null dan undefined
type MaybeString = string | null | undefined;
type DefiniteString = NonNullable<MaybeString>;
// string

// ReturnType - mengambil return type dari function
function getUser(): User {
    return { id: 1, name: "John", email: "john@example.com", age: 30, isActive: true };
}

type GetUserReturn = ReturnType<typeof getUser>;
// User

// Parameters - mengambil parameter types dari function
function createUser(name: string, email: string, age: number): User {
    return { id: Date.now(), name, email, age, isActive: true };
}

type CreateUserParams = Parameters<typeof createUser>;
// [string, string, number]

// ConstructorParameters - mengambil constructor parameter types
class Car {
    constructor(public brand: string, public model: string, public year: number) {}
}

type CarConstructorParams = ConstructorParameters<typeof Car>;
// [string, string, number]

// InstanceType - mengambil instance type dari constructor
type CarInstance = InstanceType<typeof Car>;
// Car
```

## Practical Advanced Type Examples

```typescript
// Deep object path type
type DeepKeys<T> = T extends object
    ? {
        [K in keyof T]: K extends string
            ? T[K] extends object
                ? `${K}` | `${K}.${DeepKeys<T[K]>}`
                : `${K}`
            : never;
    }[keyof T]
    : never;

interface NestedConfig {
    database: {
        host: string;
        port: number;
        credentials: {
            username: string;
            password: string;
        };
    };
    api: {
        version: string;
        timeout: number;
    };
}

type ConfigPaths = DeepKeys<NestedConfig>;
// "database" | "database.host" | "database.port" | "database.credentials" | 
// "database.credentials.username" | "database.credentials.password" | 
// "api" | "api.version" | "api.timeout"

// Type-safe event emitter
type EventMap = {
    userLogin: { userId: number; timestamp: Date };
    userLogout: { userId: number };
    dataUpdate: { table: string; recordId: number };
};

class TypedEventEmitter<T extends Record<string, any>> {
    private listeners: {
        [K in keyof T]?: Array<(data: T[K]) => void>;
    } = {};

    on<K extends keyof T>(event: K, listener: (data: T[K]) => void): void {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event]!.push(listener);
    }

    emit<K extends keyof T>(event: K, data: T[K]): void {
        const eventListeners = this.listeners[event];
        if (eventListeners) {
            eventListeners.forEach(listener => listener(data));
        }
    }
}

const emitter = new TypedEventEmitter<EventMap>();

emitter.on('userLogin', (data) => {
    // data is correctly typed as { userId: number; timestamp: Date }
    console.log(`User ${data.userId} logged in at ${data.timestamp}`);
});

emitter.emit('userLogin', { userId: 123, timestamp: new Date() });

// Builder pattern dengan types
class QueryBuilder<T> {
    private conditions: string[] = [];
    private selectFields: (keyof T)[] = [];

    select<K extends keyof T>(...fields: K[]): QueryBuilder<Pick<T, K>> {
        this.selectFields = fields;
        return this as any;
    }

    where(condition: string): QueryBuilder<T> {
        this.conditions.push(condition);
        return this;
    }

    build(): string {
        const select = this.selectFields.length > 0 
            ? this.selectFields.join(', ') 
            : '*';
        const where = this.conditions.length > 0 
            ? ` WHERE ${this.conditions.join(' AND ')}`
            : '';
        
        return `SELECT ${select} FROM table${where}`;
    }
}

// Usage
const query = new QueryBuilder<User>()
    .select('name', 'email')
    .where('age > 18')
    .where('isActive = true')
    .build();

console.log(query); // "SELECT name, email FROM table WHERE age > 18 AND isActive = true"
```

Advanced types adalah kekuatan sejati TypeScript yang memungkinkan kita membuat sistem tipe yang sangat ekspresif dan aman. Dengan menguasai konsep-konsep ini, kita bisa membangun aplikasi yang robust dengan type safety yang luar biasa.