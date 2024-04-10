import LoadData
import snn
import torch
from snntorch import functional as SF


if __name__ == "__main__":

    #Load the Dataset
    trainingSet = LoadData.LoadDataset()
    trainingSet = trainingSet.float()


    trainY = trainingSet[:, -1]
    trainX = trainingSet[:, :-1]

   
    #Set Hyperparams
    layerDims = [3, 50, 10]
    beta = 0.99
    learningRate = 0.001
    epochs = 3000
   
    #Init SNN model
    modelSNN = snn.SNN(layerDims, beta)

    #init optimizer and loss function
    optimizer = torch.optim.Adam(modelSNN.parameters(), lr=learningRate)
    lossFunc = SF.mse_count_loss()


    #Training loop. Doesn't work still WIP.
    for epoch in range(0, epochs):

        outspk = modelSNN(trainX)
        outspk = outspk.flatten()
        loss = lossFunc(outspk, trainY)

        print(loss)

        loss.backward()
        optimizer.step()
    

   




    

    


    

    

