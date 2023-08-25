import torch

def training(model, optimizer, criterion, 
             nb_epochs, data_loader):
                 
    training_loss = []            
    for epoch in range(nb_epochs):
        for batch_images, batch_labels in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

        print(f"Epoch [{epoch+1}/{nb_epochs}], Loss: {loss.item():.4f}")
        
    return training_loss

def test(model, data_loader):
    correct = 0
    total = 0
    
    error_idx = []
    predicted_list = []
    with torch.no_grad():
        for batch_images, batch_labels in data_loader:
            outputs = model(batch_images)
            _, predicted = torch.max(outputs, dim=1)
            total += batch_labels.shape[0]
            correct += int((predicted == batch_labels).sum())
            error_idx.append(predicted == batch_labels)
            predicted_list.append(predicted)
            
    print(f"Test Accuracy: {(correct / total)*100 }%")

    return error_idx, predicted_list
    
        
  