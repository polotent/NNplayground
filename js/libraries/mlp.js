// MULTILAYER PERCEPTRON
class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);

class NeuralNetwork {
  // i - input, h - hidden, o - output
  constructor(iNodes, hLayers, oNodes) {
    this.iNodes = iNodes;
    this.hNodes = hLayers;
    this.hLayersNum = hLayers.length;
    this.oNodes = oNodes;

    this.weights = [];
    this.biases = [];
    //--------------INPUT->HIDDEN--------------
    let weightsMatrixNew = new Matrix(this.hNodes[0], this.iNodes);
    this.weights.push(weightsMatrixNew);
    this.weights[0].randomize();

    //--------------HIDDEN->HIDDEN-------------
    let biasesMatrixNew;
    for (let i = 0; i < this.hLayersNum - 1; i++){
      //generating weights
      weightsMatrixNew = new Matrix(this.hNodes[i + 1], this.hNodes[i]);
      this.weights.push(weightsMatrixNew);
      this.weights[i + 1].randomize();
      //generating biases
      biasesMatrixNew = new Matrix(this.hNodes[i], 1);
      this.biases.push(biasesMatrixNew);
      this.biases[i].randomize();
    }
    //generating last hidden layer bias matrix
    biasesMatrixNew = new Matrix(this.hNodes[this.hLayersNum - 1], 1);
    this.biases.push(biasesMatrixNew);
    this.biases[this.hLayersNum - 1].randomize();

    //--------------HIDDEN->OUTPUT-------------
    weightsMatrixNew = new Matrix(this.oNodes, this.hNodes[this.hLayersNum - 1]);
    this.weights.push(weightsMatrixNew);
    this.weights[this.hLayersNum].randomize();
    biasesMatrixNew = new Matrix(this.oNodes, 1);
    this.biases.push(biasesMatrixNew);
    this.biases[this.hLayersNum].randomize();
    //Set options
    this.setLearningRate();
    this.setActivationFunction();
  }

  setLearningRate(learningRate = 0.1) {
    this.learningRate = learningRate;
  }

  setActivationFunction(func = sigmoid) {
    this.activationFunction = func;
  }

  predict(inputArray) {
    //-------------FEEDFORWARD---------------
    let buffArray = Matrix.fromArray(inputArray);
    for (let i = 0; i < (this.hLayersNum - 1) + 2; i++){
      buffArray = Matrix.multiply(this.weights[i], buffArray);
      buffArray.add(this.biases[i]);
      buffArray.map(this.activationFunction.func);
    }
    let outputs = buffArray.toArray();
    // Returning the output vector
    return outputs;
  }

  train(inputArray, targetArray) {
    //-------------FEEDFORWARD---------------
    let buffArray = Matrix.fromArray(inputArray);
    let results = [];
    results.push(buffArray);
    for (let i = 0; i < (this.hLayersNum - 1) + 2; i++){
      buffArray = Matrix.multiply(this.weights[i], buffArray);
      buffArray.add(this.biases[i]);
      buffArray.map(this.activationFunction.func);
      results.push(buffArray);
    }

    //----------BACKPROPOGATION ERROR---------------
    //--------ERROR = [TARGETS - OUTPUTS]-----------
    let targets, errors, gradients, prev_targets, deltas;
    for (let i = results.length - 1; i > 0; i--){
      if (i == results.length - 1) {
        targets = Matrix.fromArray(targetArray);
        // Calculate errors
        errors = Matrix.subtract(targets, results[i]);
      } else {
        targets = Matrix.transpose(this.weights[i]);
        // Calculate errors
        errors = Matrix.multiply(targets, errors);
      }

      // Calculate gradient
      gradients = Matrix.map(results[i], this.activationFunction.dfunc);
      gradients.multiply(errors);
      gradients.multiply(this.learningRate);

      // Calculate deltas
      prev_targets = Matrix.transpose(results[i-1]);
      deltas = Matrix.multiply(gradients, prev_targets);

      // Adjust the weights by deltas
      this.weights[i - 1].add(deltas);
      // Adjust the bias by its deltas (which is just the gradients)
      this.biases[i - 1].add(gradients);
    }
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == 'string') {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(data.iNodes, data.hNodes, data.oNodes);

    nn.weights = [];
    nn.biases = [];
    for (let i = 0; i < data.hLayersNum + 1; i++){
      nn.weights.push(Matrix.deserialize(data.weights[i]));
      nn.biases.push(Matrix.deserialize(data.biases[i]));
    }
    nn.learningRate = data.learningRate;
    return nn;
  }
}
