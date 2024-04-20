import YCData
import TrainData
import models
import torch
choices=["mlp","bi-lstm","rnn","gnn","all"]
s,l=YCData.get_sequences()
classes=max(l)+1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def run(n):
    match n:
        case 1:
            print("MLP model")
            print("processing data for MLP")
            train,x_test,y_test=TrainData.data_mlp(s,l)
            model = models.MLP(classes,100, classes).to(device)
            print("Training MLP")
            model.fit(train)
            print("Testing MLP")
            model.test(x_test,y_test)
        case 2:
            print("Bi-LSTM model")
            print("processing data for Bi-LSTM")
            train,test=TrainData.data_rnn(s,l)
            embedding_dim = 100
            hidden_dim = 256
            output_dim = classes
            num_layers = 2
            model = models.Bi_LSTM(classes, embedding_dim, hidden_dim, output_dim, num_layers)
            model.to(device)
            print("Training Bi-LSTM")
            model.fit(train)
            print("Testing Bi-LSTM")
            model.test(test)
        case 3:
            print("RNN model")
            print("processing data for RNN")
            train,test=TrainData.data_rnn(s,l)
            embedding_dim = 64
            hidden_dim = 128
            output_dim = classes
            model = models.RNNModel(classes, embedding_dim, hidden_dim, output_dim)
            model.to(device)
            print("Training RNN")
            model.fit(train)
            print("Testing RNN")
            model.test(test)
        case 4:
            print("GNN model")
            print("processing data for GNN")
            train,test=TrainData.data_gnn(s,l)
            model = models.GNN(classes,1,100,128, classes)
            model.cuda()
            print("Training GNN")
            model.fit(train)
            print("Testing GNN")
            model.test(test)
def main():
    print("Choose a model from the following:")
    print("1: MLP (Multi-Layer Perceptron)")
    print("2: Bi-LSTM (bidirectional Long Short-Term Memory)")
    print("3: RNN (Recurrent Neural Network)")
    print("4: GNN (Graph Neural Network)")
    print("5: all")
    choice = input("Enter your choice (1-5): ")
    try: 
        choice= int(choice)
        if(choice <= 5):
            if(choice==5):
                for i in range(4):
                    run(i+1)
            else:
                run(choice)
        else:print("out of range! please choose between 1 and 5.")

    except:
        if(choice.lower() in choices):
            choice=choices.index(choice.lower())+1
            print(choice)
        else:
            print("wrong input")
if __name__ == "__main__":
    main()