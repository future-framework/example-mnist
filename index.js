const f = require('future-framework')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const data = tf.tidy(() => {
  const coeff = {a: -.8, b: -.2, c: .9, d: .5};
  const numPoints = 100;
  const sigma = 0.04;

  const [a, b, c, d] = [
    tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
    tf.scalar(coeff.d)
  ];

  const xs = tf.randomUniform([numPoints], -1, 1);

  // Generate polynomial data
  const three = tf.scalar(3, 'int32');
  const ys = a.mul(xs.pow(three))
    .add(b.mul(xs.square()))
    .add(c.mul(xs))
    .add(d)
    // Add random noise to the generated data
    // to make the problem a bit more interesting
    .add(tf.randomNormal([numPoints], 0, sigma));

  // Normalize the y values to the range 0 to 1.
  const ymin = ys.min();
  const ymax = ys.max();
  const yrange = ymax.sub(ymin);
  const ysNormalized = ys.sub(ymin).div(yrange);
  console.log(xs)

  return {
    xs,
    ys: ysNormalized
  };
})
console.log(data)

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

const numIterations = 75;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3, 'int32')))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

function loss(prediction, labels) {
  // Having a good error function is key for training a machine learning model
  const error = prediction.sub(labels).square().mean();
  return error;
}

const train = f(async () => {
   for (let iter = 0; iter < numIterations; iter++) {
    // optimizer.minimize is where the training happens.

    // The function it takes must return a numerical estimate (i.e. loss)
    // of how well we are doing using the current state of
    // the variables we created at the start.

    // This optimizer does the 'backward' step of our training process
    // updating variables defined previously in order to minimize the
    // loss.
    optimizer.minimize(() => {
      // Feed the examples into the model
      const pred = predict(data.xs);
      return loss(pred, data.ys);
    });

    // Use tf.nextFrame to not block the browser.
    // await tf.nextFrame();
  }

  return {
    hello: 'Dude',
  };
}, {
  name: 'train',
  input: {},
  output: {
    hello: 'String',
  },
});

const predictNumbers = f(async () => {
  return {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  };
}, {
  name: 'predictNumbers',
  input: {},
  output: {
    a: 'Float',
    b: 'Float',
    c: 'Float',
    d: 'Float',
  },
});

const run = async () => {
  await train();
  console.log(await predictNumbers());
};

run();
