const doTraining = async model => {
  const history = await model.fit(xs, ys, {
    epochs: 500, // number of training iterations
      callbacks: {
        onEpochEnd: async(epoch, logs) => {
          console.log("Epoch:" + epoch + " Loss:" + logs.loss);
        }
      }
  });
}

const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss:'meanSquaredError', optimizer:'sgd'});
model.summary()

const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]); // array of inputs (x)
const ys = tf.tensor2d([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], [6, 1]); // array of correct answers (results - y)

doTraining(model).then(() => {
  alert(model.predict(tf.tensor2d([10], [1,1]))); // we want to know the prediction for input x = 10
});