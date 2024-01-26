export default class NeuralNetwork {
    private inputSize: number;
    private hiddenSize: number;
    private outputSize: number;

    private weightsInputToHidden: number[][];
    private biasHidden: number[];

    private weightsHiddenToOutput: number[][];
    private biasOutput: number[];

    private learningRate: number;

    constructor(inputSize: number, hiddenSize: number, outputSize: number, learningRate?: number) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.weightsInputToHidden = Array.from({ length: hiddenSize }, () =>
            Array.from({ length: inputSize }, () => Math.random() * 2 - 1)
        );
        this.biasHidden = Array(hiddenSize).fill(0);
        this.weightsHiddenToOutput = Array.from({ length: outputSize }, () =>
            Array.from({ length: hiddenSize }, () => Math.random() * 2 - 1)
        );
        this.biasOutput = Array(outputSize).fill(0);
        this.learningRate = learningRate || 0.1;
    }

    predict(inputs: number[]) {
        const hiddenLayer = new Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
            hiddenLayer[i] = 0;
            for (let j = 0; j < this.inputSize; j++) {
                hiddenLayer[i] += this.weightsInputToHidden[i][j] * inputs[j];
            }
            hiddenLayer[i] += this.biasHidden[i];
            hiddenLayer[i] = this.sigmoid(hiddenLayer[i]);
        }

        const output = new Array(this.outputSize);
        for (let i = 0; i < this.outputSize; i++) {
            output[i] = 0;
            for (let j = 0; j < this.hiddenSize; j++) {
                output[i] +=
                    this.weightsHiddenToOutput[i][j] * hiddenLayer[j];
            }
            output[i] += this.biasOutput[i];
            output[i] = this.sigmoid(output[i]);
        }
        return output;
    }

    train(inputs: number[], target: number[]) {
        const hiddenLayer = new Array(this.hiddenSize);
        for (let i = 0; i < this.hiddenSize; i++) {
            hiddenLayer[i] = 0;
            for (let j = 0; j < this.inputSize; j++) {
                hiddenLayer[i] +=
                    this.weightsInputToHidden[i][j] * inputs[j];
            }
            hiddenLayer[i] += this.biasHidden[i];
            hiddenLayer[i] = this.sigmoid(hiddenLayer[i]);
        }

        const output = new Array(this.outputSize);
        for (let i = 0; i < this.outputSize; i++) {
            output[i] = 0;
            for (let j = 0; j < this.hiddenSize; j++) {
                output[i] += this.weightsHiddenToOutput[i][j] * hiddenLayer[j];
            }
            output[i] += this.biasOutput[i];
            output[i] = this.sigmoid(output[i]);
        }


        const errorsOutput = this.CalculateMSE(output, target);

        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.weightsHiddenToOutput[i][j] +=
                    this.learningRate *
                    errorsOutput
                    output[i] *
                    (1 - output[i]) *
                    hiddenLayer[j];
            }
            this.biasOutput[i] = this.learningRate * errorsOutput;
        }

        const errorsHidden = new Array(this.hiddenSize);

        for (let i = 0; i < this.hiddenSize; i++) {
            errorsHidden[i] = 0;
            
            for (let j = 0; j < this.outputSize; j++) {
                errorsHidden[i] += this.weightsHiddenToOutput[j][i] * errorsOutput;
            }

            this.biasHidden[i] += this.learningRate * errorsHidden[i];
            for (let j = 0; j < this.inputSize; j++) {
                this.weightsInputToHidden[i][j] +=
                    this.learningRate *
                    errorsHidden[i] *
                    hiddenLayer[i] *
                    (1 - hiddenLayer[i]) *
                    inputs[j];
            }
        }
    }

    sigmoid(x: number) {
        return 1 / (1 + Math.exp(-x));
    }

    CalculateMSE(targets: number[], outputs: number[]) {
        let mse = 0;
        
        for (let i = 0; i < targets.length; i++) {
            mse = (targets[i] - outputs[i]) ** 2
        }

        mse = mse / targets.length
        
        if (isNaN(mse)) {
            return 0;
        }
    
        return mse;
    }
}