# [typescript] #04 - TS Classes

![ts-fundamental](https://raw.githubusercontent.com/oujisan/OuVault/main/img/ts.png)

## Basic Class Syntax
---
TypeScript classes menambahkan fitur type safety dan access modifiers ke ES6 classes.

```typescript
class Person {
    // Properties dengan types
    name: string;
    age: number;
    email: string;
    
    // Constructor
    constructor(name: string, age: number, email: string) {
        this.name = name;
        this.age = age;
        this.email = email;
    }
    
    // Methods
    introduce(): string {
        return `Hi, I'm ${this.name}, ${this.age} years old`;
    }
    
    celebrateBirthday(): void {
        this.age++;
        console.log(`Happy birthday! Now ${this.age} years old`);
    }
}

// Creating instances
const person1 = new Person("Alice", 25, "alice@example.com");
const person2 = new Person("Bob", 30, "bob@example.com");

console.log(person1.introduce()); // "Hi, I'm Alice, 25 years old"
person1.celebrateBirthday(); // "Happy birthday! Now 26 years old"
```

## Access Modifiers
---
TypeScript menyediakan tiga access modifier untuk mengontrol visibility properties dan methods.

### Public (Default)

```typescript
class Car {
    public brand: string;  // Explicit public
    model: string;         // Implicit public (default)
    
    constructor(brand: string, model: string) {
        this.brand = brand;
        this.model = model;
    }
    
    public startEngine(): void {
        console.log("Engine started");
    }
}

const car = new Car("Toyota", "Camry");
console.log(car.brand);  // ✓ Accessible
car.startEngine();       // ✓ Accessible
```

### Private

```typescript
class BankAccount {
    private balance: number;
    public accountNumber: string;
    
    constructor(accountNumber: string, initialBalance: number) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
    }
    
    private validateAmount(amount: number): boolean {
        return amount > 0 && amount <= this.balance;
    }
    
    public withdraw(amount: number): boolean {
        if (this.validateAmount(amount)) {
            this.balance -= amount;
            return true;
        }
        return false;
    }
    
    public getBalance(): number {
        return this.balance;
    }
}

const account = new BankAccount("12345", 1000);
console.log(account.getBalance()); // ✓ 1000
account.withdraw(200);             // ✓ Valid

// console.log(account.balance);      // ✗ Error: private
// account.validateAmount(100);       // ✗ Error: private
```

### Protected

```typescript
class Animal {
    protected name: string;
    protected species: string;
    
    constructor(name: string, species: string) {
        this.name = name;
        this.species = species;
    }
    
    protected makeSound(): void {
        console.log("Some generic animal sound");
    }
}

class Dog extends Animal {
    private breed: string;
    
    constructor(name: string, breed: string) {
        super(name, "Canine");  // Call parent constructor
        this.breed = breed;
    }
    
    public bark(): void {
        console.log(`${this.name} barks: Woof!`);
        this.makeSound(); // ✓ Accessible in child class
    }
    
    public getInfo(): string {
        return `${this.name} is a ${this.breed} (${this.species})`;
    }
}

const dog = new Dog("Buddy", "Golden Retriever");
dog.bark();           // ✓ "Buddy barks: Woof!"
console.log(dog.getInfo()); // ✓ "Buddy is a Golden Retriever (Canine)"

// console.log(dog.name);     // ✗ Error: protected
// dog.makeSound();           // ✗ Error: protected
```

## Constructor Shorthand
---
TypeScript menyediakan syntax shorthand untuk properties dalam constructor.

```typescript
// Traditional way
class Student {
    public name: string;
    private age: number;
    protected studentId: string;
    
    constructor(name: string, age: number, studentId: string) {
        this.name = name;
        this.age = age;
        this.studentId = studentId;
    }
}

// Shorthand way (equivalent)
class StudentShort {
    constructor(
        public name: string,
        private age: number,
        protected studentId: string
    ) {
        // Properties automatically created and assigned
    }
    
    public getAge(): number {
        return this.age;
    }
}

const student = new StudentShort("John", 20, "STU001");
console.log(student.name);    // ✓ "John"
console.log(student.getAge()); // ✓ 20
```

## Getters and Setters
---
```typescript
class Temperature {
    private _celsius: number = 0;
    
    // Getter
    get celsius(): number {
        return this._celsius;
    }
    
    // Setter
    set celsius(value: number) {
        if (value < -273.15) {
            throw new Error("Temperature cannot be below absolute zero");
        }
        this._celsius = value;
    }
    
    // Computed property
    get fahrenheit(): number {
        return (this._celsius * 9/5) + 32;
    }
    
    set fahrenheit(value: number) {
        this.celsius = (value - 32) * 5/9;
    }
}

const temp = new Temperature();
temp.celsius = 25;
console.log(temp.fahrenheit); // 77

temp.fahrenheit = 100;
console.log(temp.celsius);    // 37.77777777777778
```

## Static Members
---
Static properties dan methods belong ke class itself, bukan ke instance.

```typescript
class MathHelper {
    static readonly PI: number = 3.14159;
    private static instanceCount: number = 0;
    
    constructor() {
        MathHelper.instanceCount++;
    }
    
    static calculateCircleArea(radius: number): number {
        return MathHelper.PI * radius * radius;
    }
    
    static getInstanceCount(): number {
        return MathHelper.instanceCount;
    }
    
    // Instance method bisa akses static members
    showPi(): void {
        console.log(`PI is ${MathHelper.PI}`);
    }
}

// Using static members tanpa instance
console.log(MathHelper.PI);                    // 3.14159
console.log(MathHelper.calculateCircleArea(5)); // 78.53975

// Creating instances
const helper1 = new MathHelper();
const helper2 = new MathHelper();

console.log(MathHelper.getInstanceCount()); // 2
```

## Abstract Classes
---
Abstract classes tidak bisa di-instantiate langsung dan biasanya digunakan sebagai base class.

```typescript
abstract class Shape {
    protected color: string;
    
    constructor(color: string) {
        this.color = color;
    }
    
    // Abstract method - harus diimplementasi di child class
    abstract calculateArea(): number;
    abstract getPerimeter(): number;
    
    // Concrete method - bisa digunakan langsung
    displayInfo(): void {
        console.log(`Color: ${this.color}, Area: ${this.calculateArea()}`);
    }
}

class Rectangle extends Shape {
    constructor(
        private width: number,
        private height: number,
        color: string
    ) {
        super(color);
    }
    
    calculateArea(): number {
        return this.width * this.height;
    }
    
    getPerimeter(): number {
        return 2 * (this.width + this.height);
    }
}

class Circle extends Shape {
    constructor(
        private radius: number,
        color: string
    ) {
        super(color);
    }
    
    calculateArea(): number {
        return Math.PI * this.radius * this.radius;
    }
    
    getPerimeter(): number {
        return 2 * Math.PI * this.radius;
    }
}

// const shape = new Shape("red"); // ✗ Error: Cannot instantiate abstract class

const rectangle = new Rectangle(10, 5, "blue");
const circle = new Circle(3, "red");

rectangle.displayInfo(); // "Color: blue, Area: 50"
circle.displayInfo();    // "Color: red, Area: 28.274333882308138"
```

## Implementing Interfaces
---
Classes bisa mengimplementasi satu atau lebih interfaces.

```typescript
interface Flyable {
    fly(): void;
    altitude: number;
}

interface Swimmable {
    swim(): void;
    depth: number;
}

class Duck implements Flyable, Swimmable {
    altitude: number = 0;
    depth: number = 0;
    
    constructor(private name: string) {}
    
    fly(): void {
        this.altitude = 100;
        console.log(`${this.name} is flying at ${this.altitude}m`);
    }
    
    swim(): void {
        this.depth = 2;
        console.log(`${this.name} is swimming at ${this.depth}m depth`);
    }
    
    quack(): void {
        console.log(`${this.name} says: Quack!`);
    }
}

const duck = new Duck("Donald");
duck.fly();   // "Donald is flying at 100m"
duck.swim();  // "Donald is swimming at 2m depth"
duck.quack(); // "Donald says: Quack!"
```

## Generic Classes
---
Classes bisa menggunakan generics untuk type flexibility.

```typescript
class Container<T> {
    private items: T[] = [];
    
    add(item: T): void {
        this.items.push(item);
    }
    
    get(index: number): T | undefined {
        return this.items[index];
    }
    
    getAll(): T[] {
        return [...this.items]; // Return copy
    }
    
    size(): number {
        return this.items.length;
    }
    
    clear(): void {
        this.items = [];
    }
}

// String container
const stringContainer = new Container<string>();
stringContainer.add("hello");
stringContainer.add("world");
console.log(stringContainer.getAll()); // ["hello", "world"]

// Number container
const numberContainer = new Container<number>();
numberContainer.add(1);
numberContainer.add(2);
console.log(numberContainer.getAll()); // [1, 2]

// Custom type container
interface Product {
    id: number;
    name: string;
    price: number;
}

const productContainer = new Container<Product>();
productContainer.add({ id: 1, name: "Laptop", price: 1000 });
productContainer.add({ id: 2, name: "Mouse", price: 25 });
```

Classes dalam TypeScript memberikan structure yang kuat untuk object-oriented programming dengan type safety yang sangat baik. Mereka sangat berguna untuk modeling complex business logic dan creating reusable components.