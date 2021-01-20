const fs = require("fs");
const faker = require("faker");

const MANUFACTURERS = [];

for (let i = 0; i < 10; i++) {
  MANUFACTURERS.push({ id: i, name: faker.vehicle.manufacturer() });
}

console.log(MANUFACTURERS);

fs.writeFileSync("manufacturers.json", manufacturers);
