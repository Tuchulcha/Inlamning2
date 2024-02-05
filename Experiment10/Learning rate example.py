# Learning rate changer function
def learning_rate(epoch, total_epochs):
    if epoch < total_epochs / 4:
        lr = 0.001 * (epoch / (total_epochs / 4))
    elif epoch < total_epochs * (3 / 4):
        lr = 0.001
    else:
        # Calculate the remaining fraction of epochs and use it to linearly interpolate between 0.001 and 0.0005.
        remaining_epochs = epoch - total_epochs * (3 / 4)
        total_decreasing_epochs = total_epochs - total_epochs * (3 / 4)
        lr = 0.001 - (0.0005 * (remaining_epochs / total_decreasing_epochs))
    return lr

# Training function
# Adjusted training function to emphasize scheduler's role
def train_model(model, train_loader, criterion, optimizer, scheduler, total_epochs):
    model.train()
    for epoch in range(total_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # This is changing the learning rate. 
            for param_group in optimizer.param_groups: # Look up what does actually does. IS the learning rate saved in many places?
                param_group['lr'] = learning_rate(epoch, total_epochs)  # Update the learning rate
        
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{total_epochs}, Loss: {total_loss/len(train_loader)}, LR: {current_lr}')