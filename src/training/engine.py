import tensorflow as tf


def train(
    dataset: tf.data.Dataset,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss,
    epochs: int,
    steps_per_epoch: int,
    log_interval: int = 10,
):
    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x)
                loss = loss_fn(y, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % log_interval == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.numpy()}")
