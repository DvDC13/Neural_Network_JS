let training_data = [
  {
  	inputs: [0, 1],
  	targets: [1]
  },
    {
  	inputs: [1, 0],
  	targets: [1]
  },
    {
  	inputs: [0, 0],
  	targets: [0]
  },
    {
  	inputs: [1, 1],
  	targets: [0]
  }
];

let nn;

function setup() {
  createCanvas(windowWidth, windowHeight);
  nn = new NeuralNetwork(2, 2, 1);
  for (let i = 0; i < 500000; i++) {
    let data = random(training_data);
    nn.train(data.inputs, data.targets);
  }

  console.log(nn.feedforward([1, 0]));
  console.log(nn.feedforward([0, 1]));
  console.log(nn.feedforward([1, 1]));
  console.log(nn.feedforward([0, 0]));

}

function draw() {
  for (let i = 0; i < 1000; i++) {
  	let data = random(training_data);
    nn.train(data.inputs, data.targets);
  }

  let resolution = 10;
  let cols = width / resolution;
  let rows = height / resolution;
  for (let i = 0; i < cols; i++){
  	for (let j = 0; j < rows; j++) {
  		let x1 = i / cols;
  		let x2 = j / rows;
  		let inputs = [x1, x2]; 
  		let y = nn.feedforward(inputs);
  		noStroke();
  		fill(y * 255);
  		rect(i * resolution, j * resolution, resolution, resolution);
  	}
  }
}