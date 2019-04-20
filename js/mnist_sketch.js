const len = 784;

let mnist_dataset = {};
let files = {};

let nnMain;
let nnGuess;

let mnistStuff;
let qdStuff;

let cnv;

let k = false;

let updatePage = false;
let resetPage = false;
let currentPath = "";
let currentFilename = "";
let currentZone = "";

let qdLabelList = ["apple", "bed", "bridge",
                "cactuse", "eye", "ladder",
                "scissors", "star", "sun", "tshirt"];

let qd_dataset = {};

let els = {
  "bars" : {},
  "buttons" : {},
  "labels" : {},
  "charts" : {},
  "contexts" : {},
  "forms" : {},
  "inputs" : {}
}

let stuff = {
  "trainQuantity" : 5500,
  "validationQuantity" : 500,
  "testQuantity" : 1000,
  "state" : "training",
  "active" : false,
  "index" : 0,
  "progress" : 0,
  "correct" : 0,
  "accuracy" : 0,
  "epochCounter" : 1,
  "maxEpoch" : 10,
  "testErrors" : [90],
  "validationErrors" : [90],
  "labels" : [0],
  "minError" : 2,
  "drawn" : false,
  "pos" : {
    "x" : -1,
    "y" : -1
  },
  "states" : {}
};

function showPage() {
  document.getElementById("loader").style.display = "none";
  document.getElementById("mainDiv").style.display = "block";
}

function preload() {
  files.images60k = loadMNIST('data/mnist/train-images.idx3-ubyte', 16);
  files.labels60k = loadMNIST('data/mnist/train-labels.idx1-ubyte', 8);
  files.images10k = loadMNIST('data/mnist/t10k-images.idx3-ubyte', 16);
  files.labels10k = loadMNIST('data/mnist/t10k-labels.idx1-ubyte', 8);

  qdLabelList.forEach(function(item){
    let buff = item + "10k";
    files[buff] = loadQD('data/quickdraw/' + item + '10k.bin');
  });

  nn100mnist = loadJSON("./models/mnist100.json");
  nn100qd = loadJSON("./models/quickdraw100.json");
}

function prepareData() {
  mnist_dataset.training = [];
  mnist_dataset.testing = [];
  mnist_dataset.validating = [];
  for (let i = 0; i < 60000; i++){
    let offset = i * len;
    if (i < 55000){
      mnist_dataset.training[i] = files.images60k.bytes.subarray(offset, offset + len);
      mnist_dataset.training[i].label = files.labels60k.bytes[i];
    } else {
      mnist_dataset.validating[i - 55000] = files.images60k.bytes.subarray(offset, offset + len);
      mnist_dataset.validating[i - 55000].label = files.labels60k.bytes[i];
    }
  }
  for (let i = 0; i < 10000; i++){
    let offset = i * len;
    mnist_dataset.testing[i] = files.images10k.bytes.subarray(offset, offset + len);
    mnist_dataset.testing[i].label = files.labels10k.bytes[i];
  }
  console.log(mnist_dataset);

  qd_dataset.training = [];
  qd_dataset.testing = [];
  qd_dataset.validating = [];
  let cnt = 0;
  qdLabelList.forEach(function(item){
    for (let i = 0; i < 7000; i++){
      let offset = i * len;
      let buff = item + "10k";
      if (i < 5500){
        qd_dataset.training[i + cnt * 5500] = files[buff].bytes.subarray(offset, offset + len);
        qd_dataset.training[i + cnt * 5500].label = item;
      }
      if ((i >= 5500) && (i < 6000)){
        qd_dataset.validating[i - 5500 + cnt * 500] = files[buff].bytes.subarray(offset, offset + len);
        qd_dataset.validating[i - 5500 + cnt * 500].label = item;
      }
      if (i >= 6000) {
        qd_dataset.testing[i - 6000 + cnt * 1000] = files[buff].bytes.subarray(offset, offset + len);
        qd_dataset.testing[i - 6000 + cnt * 1000].label = item;
      }
    }
    cnt = cnt + 1;
  });

  shuffle(qd_dataset.training, true);
  shuffle(qd_dataset.validating, true);
  shuffle(qd_dataset.testing, true);

  console.log(qd_dataset);

  showPage();
}

function train(training){
  let inputs = [];
  let targets = [];
  for (let j = 0; j < 784; j++) {
    let bright = training[stuff.index][j];
    inputs[j] = bright / 255;
  }
  let label = training[stuff.index].label;
  for (let j = 0; j < 10; j++) {
    targets.push(0);
  }

  if (currentZone == "MNIST") {
    targets[label] = 1;
  } else {
    let indx = qdLabelList.indexOf(label);
    targets[indx] = 1;
  }

  nnMain.train(inputs, targets);

  stuff.progress = ((stuff.index + 1) / stuff.trainQuantity) * 100;

  els.bars.progress.html(nf(stuff.progress, 2, 2) + '%');
  els.bars.progress.style('width', nf(stuff.progress, 2, 2) + '%');

  stuff.index++;
  if (stuff.index == stuff.trainQuantity){
    stuff.index = 0;
    stuff.state = "validating";
    els.labels.statusLabel.html('Validating');
  }
}

function validate(validating){
  let inputs = [];
  for (let j = 0; j < 784; j++) {
    let bright = validating[stuff.index][j];
    inputs[j] = bright / 255;
  }
  let label = validating[stuff.index].label;

  let guess = nnMain.predict(inputs);
  let m = max(guess);
  let classification = guess.indexOf(m);

  if (currentZone == "MNIST") {
    if (classification === label){
      stuff.correct++;
    }
  } else {
    let indx = qdLabelList.indexOf(label);
    if (classification == indx){
      stuff.correct++;
    }
  }

  stuff.progress = ((stuff.index + 1) / stuff.validationQuantity) * 100;
  stuff.accuracy = (stuff.correct / stuff.index) * 100;

  els.bars.progress.html(nf(stuff.progress, 2, 2) + '%');
  els.bars.progress.style('width', nf(stuff.progress, 2, 2) + '%');

  stuff.index++;
  if (stuff.index == stuff.validationQuantity){

    stuff.validationErrors.push(nf(100-stuff.accuracy, 2, 2));

    stuff.labels.push(stuff.epochCounter.toString());
    setGraph();

    stuff.progress = 0;
    stuff.correct = 0;
    stuff.accuracy = 0;
    stuff.index = 0;
    stuff.state = "testing";

    els.labels.statusLabel.html('Testing');
  }
}

function test(testing){
  let inputs = [];
  for (let j = 0; j < 784; j++) {
    let bright = testing[stuff.index][j];
    inputs[j] = bright / 255;
  }
  let label = testing[stuff.index].label;

  let guess = nnMain.predict(inputs);
  let m = max(guess);
  let classification = guess.indexOf(m);
  if (currentZone == "MNIST") {
    if (classification === label){
      stuff.correct++;
    }
  } else {
    let indx = qdLabelList.indexOf(label);
    if (classification == indx){
      stuff.correct++;
    }
  }

  stuff.progress = ((stuff.index + 1) / stuff.testQuantity) * 100;
  stuff.accuracy = (stuff.correct / stuff.index) * 100;

  els.bars.progress.html(nf(stuff.progress, 2, 2) + '%');
  els.bars.progress.style('width', nf(stuff.progress, 2, 2) + '%');

  stuff.index++;
  if (stuff.index == stuff.testQuantity){
    stuff.testErrors.push(nf(100-stuff.accuracy, 2, 2));

    setGraph();

    let buff_str = "Epoch " + stuff.epochCounter.toString();
    stuff.states[buff_str] = {};
    stuff.states[buff_str].state = NeuralNetwork.deserialize(nnMain.serialize());
    stuff.states[buff_str].accuracy = stuff.accuracy;

    els.forms.selectState.option(buff_str + " ( accuracy : " + nf(stuff.accuracy, 2, 2) + "% )");

    stuff.epochCounter++;
    stuff.correct = 0;
    stuff.index = 0;
    stuff.state = "training";

    let ln = stuff.validationErrors.length;
    if (stuff.epochCounter > stuff.maxEpoch) {
      stuff.state = "finished";
      els.labels.statusLabel.html('Finished (max set epoch reached)');

      els.buttons.control.style('display','none');
    } else if (stuff.minError >= stuff.validationErrors[ln-1]) {
      stuff.state = "finished";
      els.labels.statusLabel.html('Finished (required error reached)');

      els.buttons.control.style('display','none');
    } else {
      els.labels.statusLabel.html('Training');
      els.labels.statusId.html(stuff.epochCounter);
    }
    //shuffling all dataset
    if (currentZone == "MNIST") {
      shuffle(mnist_dataset.training, true);
    } else {
      shuffle(qd_dataset.training, true);
    }
  }
}

function setup(){

  cnv = createCanvas(280, 280).parent('guessing-canvas');
  pixelDensity(1);

  background(0);

  prepareData();

  nnMain = new NeuralNetwork(len, [100], 10);
  nnMain.setLearningRate(0.2);
  stuff.states["Epoch 0"] = {};
  stuff.states["Epoch 0"].state = NeuralNetwork.deserialize(nnMain.serialize());
  stuff.states["Epoch 0"].accuracy = "10.00%";
  nnGuess = stuff.states["Epoch 0"].state;

  //LABELS
  els.labels.changeNotification = select("#change-notification");

  // SELECT
  els.forms.selectState = select("#select-state");
  els.forms.selectState.option("Epoch 0 ( accuracy : 10.00% )");
  els.forms.selectState.changed(selectStateChanged);

  els.forms.selectZone = select('#select-zone');
  els.forms.selectZone.changed(selectZoneChanged);
  currentZone = els.forms.selectZone.value();

  // INPUTS
  els.inputs.modelName = select("#model-name");

  els.inputs.structure = select("#config-structure");
  els.inputs.lr = select("#config-lr");
  els.inputs.trainSize = select("#config-train-size");
  els.inputs.validateSize = select("#config-validate-size");
  els.inputs.testSize = select("#config-test-size");

  // CONTROL BUTTON
  els.buttons.control = select("#control");
  els.bars.progress = select("#progress");
  els.buttons.control.mousePressed(function() {
    if (stuff.active == false){
      stuff.active = true;
      els.buttons.control.html('STOP');
      els.buttons.control.class('btn btn-danger');
    }  else {
      stuff.active = false;
      els.buttons.control.html('CONTINUE');
      els.buttons.control.class('btn btn-warning');
    }
  });

  // CLEAR BUTTON
  els.buttons.clear = select("#clear");
  els.buttons.clear.mousePressed(function() {
    els.labels.advice.html('DRAW SOMETHING!');
    els.labels.prediction.html('');
    els.labels.prediction.style('display','none');
    stuff.drawn = false;
    cnv.background(0);
    stuff.pos.x = -1;
    stuff.pos.y = -1;
  });

  // SAVE BUTTON
  els.buttons.save = select("#save");
  els.buttons.save.mousePressed(function(){
    let blob = new Blob([JSON.stringify(stuff)], {type : "application/json"});
    saveAs(blob, els.inputs.modelName.value() + ".json");
  });


  // MNIST AND QD BUTTONS LOAD MODELS
  els.buttons.mnistLoad = select("#mnist100");
  els.buttons.mnistLoad.mousePressed(function(){
    let buffBool = stuff.drawn;
    stuff = nn100mnist;
    updatePage = true;
    els.labels.filename.html("mnist100");
    stuff.drawn = buffBool;
  });

  els.buttons.qdLoad = select("#quickdraw100");
  els.buttons.qdLoad.mousePressed(function(){
    let buffBool = stuff.drawn;
    stuff = nn100qd;
    updatePage = true;
    els.labels.filename.html("quickdraw100");
    stuff.drawn = buffBool;
  });

  //RESET BUTTON
  els.buttons.reset = select("#reset");
  els.buttons.reset.mousePressed(function(){
    resetPage = true;
  });

  //SUBMIT BUTTON
  els.buttons.submit = select("#submit");
  els.buttons.submit.mousePressed(function(){
    els.labels.changeNotification.style("display","block");

    stuff.trainQuantity = parseInt(els.inputs.trainSize.value());
    stuff.validationQuantity = parseInt(els.inputs.validateSize.value());
    stuff.testQuantity = parseInt(els.inputs.testSize.value());

    let val = els.inputs.structure.value();
    let arr = val.split(",").map(Number);
    nnMain = new NeuralNetwork(len, arr, 10);
    nnMain.setLearningRate(parseFloat(els.inputs.lr.value()));
    stuff.states["Epoch 0"] = {};
    stuff.states["Epoch 0"].state = NeuralNetwork.deserialize(nnMain.serialize());
    stuff.states["Epoch 0"].accuracy = "10.00%";
    nnGuess = stuff.states["Epoch 0"].state;
  });



  $('#files').change(function(event){
    currentPath = URL.createObjectURL(event.target.files[0]);
    currentFilename = event.target.files[0].name;

    if (currentPath != ""){
      let buffBool = stuff.drawn;
      stuff = loadJSON(currentPath, function(){
        updatePage = true;
        let buff = currentFilename.substring(0, currentFilename.lastIndexOf('.'));
        els.labels.filename.html(buff);
        stuff.drawn = buffBool;
      }, function(response){
        els.labels.filename.html("File read error!");
      });
    }
  })

  // LABELS
  els.labels.statusLabel = select("#status-label");
  els.labels.statusId = select("#status-id");

  els.labels.prediction = select("#prediction");
  els.labels.advice = select("#advice");
  els.labels.itemList = select("#item-list");
  if (currentZone == "MNIST"){
    els.labels.itemList.html("[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]");
  } else {
    els.labels.itemList.html("[ apple, bed, bridge, cactuse, eye, ladder, scissors, star, sun, tshirt ]");
  }

  els.labels.filename = select("#filename");

  els.contexts.ctx = document.getElementById('error-chart').getContext('2d');
  setGraph();
}

function draw(){
  if (stuff.active){
    if (stuff.state == "training"){
      if (currentZone == "MNIST"){
        train(mnist_dataset.training);
      } else {
        train(qd_dataset.training);
      }
    }
    if (stuff.state == "validating") {
      if (currentZone == "MNIST"){
        validate(mnist_dataset.validating);
      } else {
        validate(qd_dataset.validating);
      }
    }
    if (stuff.state == "testing"){
      if (currentZone == "MNIST"){
        test(mnist_dataset.testing);
      } else {
        test(qd_dataset.testing);
      }
    }
  }

  if (updatePage){
    updatePage = false;
    els.bars.progress.html(nf(stuff.progress, 2, 2) + '%');
    els.bars.progress.style('width', nf(stuff.progress, 2, 2) + '%');
    if (stuff.state == "training"){
      els.labels.statusLabel.html("Training");
    }
    if (stuff.state == "validating") {
      els.labels.statusLabel.html("Validating");
    }
    if (stuff.state == "testing"){
      els.labels.statusLabel.html("Testing");
    }
    if (stuff.state == "finished"){
      if (stuff.epochCounter > stuff.maxEpoch) {
        els.labels.statusLabel.html('Finished (max set epoch reached)');
        els.buttons.control.style('display','none');
      } else {
        els.labels.statusLabel.html('Finished (required error reached)');
        els.buttons.control.style('display','none');
      }
    }
    els.labels.statusId.html(stuff.epochCounter-1);
    setGraph();
    while (els.forms.selectState.elt.options.length > 0) {
      els.forms.selectState.elt.options[0].remove(0);
    }
    for (let i = 0; i < stuff.testErrors.length; i++){
      let buff_str = "Epoch " + i;
      els.forms.selectState.option(buff_str + " ( accuracy : " + nf(100 - stuff.testErrors[i], 2, 2) + "% )");
    }

    els.labels.advice.html('DRAW SOMETHING!');
    els.labels.prediction.html('');
    els.labels.prediction.style('display','none');
    // stuff.drawn = false;
    // cnv.background(0);
    // stuff.pos.x = -1;
    // stuff.pos.y = -1;

    let buff = stuff.testErrors.length - 1;
    nnMain = NeuralNetwork.deserialize(stuff.states["Epoch " + buff].state);
    nnGuess = NeuralNetwork.deserialize(stuff.states["Epoch 0"].state);

    els.inputs.structure.value(nnMain.hNodes);
    els.inputs.lr.value(nnMain.learningRate);
    els.inputs.trainSize.value(stuff.trainQuantity);
    els.inputs.validateSize.value(stuff.validationQuantity);
    els.inputs.testSize.value(stuff.testQuantity);

  }

  if (resetPage) {
    resetPage = false;

    nnMain.learningRate = 0.2;
    stuff.trainQuantity = 55000;
    stuff.validationQuantity = 5000;
    stuff.testQuantity = 10000;

    els.labels.filename.html('UNTITLED');
    nnMain = new NeuralNetwork(len, [100], 10);
    nnMain.setLearningRate(0.2);
    stuff.states["Epoch 0"] = {};
    stuff.states["Epoch 0"].state = NeuralNetwork.deserialize(nnMain.serialize());
    stuff.states["Epoch 0"].accuracy = "10.00%";
    nnGuess = stuff.states["Epoch 0"].state;

    stuff.state = "training";
    stuff.active = false;
    stuff.index = 0;
    stuff.progress = 0,
    stuff.correct = 0,
    stuff.accuracy = 0,
    stuff.epochCounter = 1,
    stuff.testErrors = [90],
    stuff.validationErrors = [90],
    stuff.labels = [0],

    els.inputs.structure.value(nnMain.hNodes);
    els.inputs.lr.value(nnMain.learningRate);
    els.inputs.trainSize.value(stuff.trainQuantity);
    els.inputs.validateSize.value(stuff.validationQuantity);
    els.inputs.testSize.value(stuff.testQuantity);


    els.buttons.control.style('display','inline');
    els.buttons.control.html('START');
    els.buttons.control.class('btn btn-success');

    els.labels.statusLabel.html("Training");
    els.labels.statusId.html(stuff.epochCounter);

    els.bars.progress.html(nf(stuff.progress, 2, 2) + '%');
    els.bars.progress.style('width', nf(stuff.progress, 2, 2) + '%');

    while (els.forms.selectState.elt.options.length > 0) {
      els.forms.selectState.elt.options[0].remove(0);
    }
    for (let i = 0; i < stuff.testErrors.length; i++){
      let buff_str = "Epoch " + i;
      els.forms.selectState.option(buff_str + " ( accuracy : " + nf(100 - stuff.testErrors[i], 2, 2) + "% )");
    }

    setGraph();
  }

  if (stuff.drawn){
    guessUserDigit();
  }
  cnv.mouseMoved(function(){
    if (mouseIsPressed){
      if (stuff.pos.x == -1 && stuff.pos.y == -1){
        stuff.pos.x = mouseX;
        stuff.pos.y = mouseY;
      }
      stuff.drawn = true;
      cnv.stroke(255);
      cnv.strokeWeight(11);
      cnv.line(stuff.pos.x, stuff.pos.y, mouseX, mouseY);
      stuff.pos.x = mouseX;
      stuff.pos.y = mouseY;
    }
  });
  cnv.mouseReleased(function(){
    stuff.pos.x = -1;
    stuff.pos.y = -1;
  });
}


function selectStateChanged(){
  let item = els.forms.selectState.value();
  let arr = item.split(" ");
  let request = arr[0] + " " + arr[1];
  nnGuess = NeuralNetwork.deserialize(stuff.states[request].state);
}

function selectZoneChanged(){
  currentZone = els.forms.selectZone.value();
  if (currentZone == "MNIST"){
    els.labels.itemList.html("[ 0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 ]");
  } else {
    els.labels.itemList.html("[ apple, bed, bridge, cactuse, eye, ladder, scissors, star, sun, tshirt ]");
  }
}

function setGraph(){
  els.charts.chart = new Chart(els.contexts.ctx, {
      // The type of chart we want to create
      type: 'line',

      // The data for our dataset
      data: {
          labels: stuff.labels,
          datasets: [{
              label: "Error on the validation dataset",
              borderColor: 'rgb(34, 124, 46)',
              backgroundColor: 'transparent',
              data: stuff.validationErrors,
            }, {
              label: "Error on the test dataset",
              borderColor: 'rgb(101, 102, 201)',
              backgroundColor: 'transparent',
              data: stuff.testErrors,
          }]
      },

      // Configuration options go here
      options: {
        scales : {
          yAxes: [{
            ticks : {
              fontSize : 16,
              beginAtZero : true,
              max : 100,
              stepSize : 10,
              callback: function(value, index, values){
                return value + '%';
              }
            },
            scaleLabel : {
              fontSize : 16,
              display : true,
              labelString : 'Percentage of wrong predictions'
            }
          }],
          xAxes: [{
            ticks : {
              fontSize : 16
            },
            scaleLabel: {
              fontSize : 16,
              display : true,
              labelString : 'Epochs'
            }
          }]
        },
        animation: false
      }
  });
}

function guessUserDigit(){
  let img = cnv.get();
  if(!stuff.drawn) {
    return img;
  }
  img.loadPixels();

  //normalization
  let left = img.width;
  let right = 0;
  let bottom = 0;
  let top = img.height;
  for (let j = 0; j < img.height; j++) {
    for (let i = 0; i < img.width; i++) {
      let index = (i + j * img.width) * 4;
      if (img.pixels[index] != 0) {
        if (j > bottom) {
          bottom = j;
        }
        if (j < top) {
          top = j;
        }
        if (i < left){
          left = i;
        }
        if (i > right){
          right = i;
        }
      }
    }
  }

  let buffImg = img.get(left, top, (right - left), (bottom - top));
  buffImg.loadPixels();
  //fill with black color
  let img1 = createImage(img.width, img.height);
  img1.loadPixels();
  for (let j = 0; j < img1.height; j++) {
    for (let i = 0; i < img1.width; i++) {
      let index = (i + j * img1.width) * 4;
      img1.pixels[index] = 0;
      img1.pixels[index + 1] = 0;
      img1.pixels[index + 2] = 0;
      img1.pixels[index + 3] = 255;
    }
  }
  img1.updatePixels();
  for (let j = img1.height / 2 - buffImg.height / 2; j < img1.height / 2 + buffImg.height / 2; j++) {
    for (let i = img1.width / 2 - buffImg.width / 2; i < img1.width / 2 + buffImg.width / 2; i++) {
      let index = (i + j * img1.width) * 4;

      let buffIndex = (i - img1.width / 2 + buffImg.width / 2 + (j - img1.height / 2 + buffImg.height / 2) * buffImg.width) * 4;
      img1.pixels[index] = buffImg.pixels[buffIndex];
      img1.pixels[index + 1] = buffImg.pixels[buffIndex + 1];
      img1.pixels[index + 2] = buffImg.pixels[buffIndex + 2];
      img1.pixels[index + 3] = buffImg.pixels[buffIndex + 3];
    }
  }
  img1.updatePixels();
  img1.resize(28, 28);

  let inputs = [];
  for (let j = 0; j < 784; j++) {
    inputs[j] = img1.pixels[j * 4] / 255;
  }
  let guess = nnGuess.predict(inputs);
  let m = max(guess);
  let classification = guess.indexOf(m);

  els.labels.advice.html('I THINK IT IS : ');
  if (currentZone == "MNIST") {
    els.labels.prediction.html(classification);
  } else {
    els.labels.prediction.html(qdLabelList[classification]);
  }
  els.labels.prediction.style('display','inline');
  return img;
}
