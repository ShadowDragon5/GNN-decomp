import torch


def test(model, testloader, device) -> float:
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            x = data.x.to(device)
            y = data.y.to(device)

            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)

            out = model(x, edge_index, batch)
            pred = out.argmax(dim=1)  # Predicted labels

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total
