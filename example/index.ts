import NeuralNetwork from "../src/nn";

const net = new NeuralNetwork(2, 4, 1);

const inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

const outputs = [
    [0],
    [1],
    [1],
    [0],
]

const ephochs = 10;

for (let i = 0; i < ephochs; i++) {
    for (let i = 0; i < inputs.length; i++) {
        net.train(inputs[i], outputs[i]);
    }
}

const output = net.predict([1, 0]);

console.log(output)