def learning_rate_tester(total_epochs):
    """
    Prints the learning rate for each epoch according to the scheme:
      - From epoch 0 to total_epochs/4, linearly increase the learning rate from near 0 to 0.001.
      - From total_epochs/4 to 3*total_epochs/4, keep the learning rate constant at 0.001.
      - From 3*total_epochs/4 to total_epochs, linearly decrease the learning rate from 0.001 to 0.0005.
    """
    for epoch in range(total_epochs):
        if epoch < total_epochs / 4:
            lr = 0.001 * (epoch / (total_epochs / 4))
        elif epoch < total_epochs * (3 / 4):
            lr = 0.001
        else:
            # Calculate the remaining fraction of epochs and use it to linearly interpolate between 0.001 and 0.0005.
            remaining_epochs = epoch - total_epochs * (3 / 4)
            total_decreasing_epochs = total_epochs - total_epochs * (3 / 4)
            lr = 0.001 - (0.0005 * (remaining_epochs / total_decreasing_epochs))

        print(f'Epoch {epoch + 1}/{total_epochs}, Learning Rate: {lr}')

# Example usage:
total_epochs = 100
learning_rate_tester(total_epochs)



