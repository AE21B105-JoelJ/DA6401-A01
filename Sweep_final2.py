# Importing the necessary libraries required
import numpy as np
import source
import wandb
wandb.login()

# Importing and reshaping datatset
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
X_train,X_cv,y_train,y_cv = train_test_split(X_train,y_train,test_size=0.1,stratify=y_train)

X_train = X_train.reshape(54000,-1)/255.0
X_cv = X_cv.reshape(6000,-1)/255.0
X_test = X_test.reshape(10000,-1)/255.0
y_train = source.one_hot_numpy(y_train)
y_cv = source.one_hot_numpy(y_cv)
y_test = source.one_hot_numpy(y_test)

sweep_config = {
    "method" : "random",
    "metric" : {
        "name" : "acc_cv",
        "goal" : "maximize"
    },
    "parameters" : {
        "optimizer" : {"values" : ["sgd","mom","nag","rmsprop", "adam","nadam"]},
        "learning_rate" : {"values" : [0.01,0.001,0.0001]},
        "loss" : {"values" : ["cross_entropy"]},
        "initialization" : {"values" : ["Xavier","random"]},
        "batch_size" : {"values" : [16,32,64]},
        "weight_decay" : {"values" : [0,0.0005,0.5]},
        # Dynamic layer configuration
        "num_layers": {"values": [2, 3, 5]},  # Max 5 layers
        
        # Layer-specific parameters (up to 5 layers)
        "layer_1_neurons": {"values": [32,64,128]},
        "layer_1_activation": {"values": ["relu","tanh","sigmoid"]},
        
        "layer_2_neurons": {"values": [32,64,128]},
        "layer_2_activation": {"values": ["relu","tanh","sigmoid"]},   

        "layer_3_neurons": {"values": [32,64,128]},
        "layer_3_activation": {"values": ["relu","tanh","sigmoid"]},   

        "layer_4_neurons": {"values": [32,64,128]},
        "layer_4_activation": {"values": ["relu","tanh","sigmoid"]},   

        "layer_5_neurons": {"values": [32,64,128]},
        "layer_5_activation": {"values": ["relu","tanh","sigmoid"]},   


        "epoch" : {"values" : [5,10]}
    }
}
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def train(config = None):
    run = wandb.init(config=config)
    config = wandb.config
    run_name = f"lr_{config.learning_rate}_bs_{config.batch_size}_nlayer_{config.num_layers}_opt_{config.optimizer}_act_{config.layer_1_activation}_epoch_{config.epoch}_init_{config.initialization}"
    run.name = run_name

    # Access top-level parameters
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    loss = config.loss
    initialization = config.initialization
    batch_size = config.batch_size
    weight_decay = config.weight_decay
    epoch = config.epoch

    num_layers = config.num_layers

    # Architecture definition
    arch = [784]
    activation_sequence = []
    for i in range(1,num_layers+1):
        neurons = getattr(config, f"layer_{i}_neurons")
        activation = getattr(config, f"layer_{i}_activation")
        arch.append(neurons)
        activation_sequence.append(activation)
    
    # finalize 
    arch.append(10)
    activation_sequence.append("softmax")
    print(arch, activation_sequence)

    momentum = 0.98
    beta_rms = 0.98
    beta_1 = 0.9
    beta_2 = 0.99

    md1 = source.FeedForwardNeuralNetwork(arch=arch, activation_sequence=activation_sequence, optimizer=optimizer,
                                      learning_rate=learning_rate,weight_decay= weight_decay, loss=loss,initialization=initialization,momentum=momentum,beta_rms=beta_rms,
                                      beta_1=beta_1,beta_2=beta_2)
    
    batch_train = source.Batchloader(X_train, y_train, batch_size=batch_size,shuffle=True)
    #batch_test = source.Batchloader(X_test, y_test, batch_size=batch_size,shuffle=False)

    epochs = config.epoch
    for epoch in range(1,epochs+1):
        for X_, y_ in batch_train:
            md1.train_step(X_, y_, epoch)

        train_pred = md1.forward_call(inputs_=X_train)
        cv_pred = md1.forward_call(inputs_=X_cv)

        loss_train = source.find_loss(y_pred= train_pred, y_true =y_train, loss="cross_entropy")
        loss_cv = source.find_loss(y_pred= cv_pred, y_true = y_cv, loss = "cross_entropy")

        train_pred = md1.forward_call(inputs_=X_train,threshold=True)
        cv_pred = md1.forward_call(inputs_=X_cv,threshold=True)

        accuracy_train = source.accuracy(train_pred, y_train)
        accuracy_cv = source.accuracy(cv_pred,  y_cv)

        run.log(data={"loss_train" : loss_train, "loss_cv" : loss_cv, "acc_train" : accuracy_train, "acc_cv" : accuracy_cv})
        if epoch%1 == 0:
            print(F"Epoch {epoch} || loss_train : {loss_train} | loss_cv : {loss_cv} || acc_train : {accuracy_train} | acc_cv : {accuracy_cv} ||")

    wandb.summary["accuracy_cv"] = accuracy_cv
    # Finding the test statistics
    test_pred = md1.forward_call(inputs_=X_test,threshold=True)

    y_true_test = np.argmax(y_test, axis=1)
    y_pred_test = np.argmax(test_pred, axis=1)

    run.log(data = {"test_acc" : source.accuracy(test_pred,y_test)})

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Compute confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test)

    # Plot confusion matrix with custom colors and annotations
    plt.figure(figsize=(9,7))
    t_font = {'family' : 'Comic Sans MS','weight' : 'normal', 'size'   : 14}
    font = {'family' : 'Comic Sans MS','weight' : 'normal', 'size'   : 12}
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names,cbar=False)
    plt.title("Confusion Matrix", fontdict= t_font)
    plt.xlabel("Predicted",fontdict=font)
    plt.ylabel("True", fontdict=font)
    plt.tight_layout()
    # Log the plot to wandb
    wandb.log({"Confusion Matrix_chk": wandb.Image(plt.gcf())})
    plt.close()


sweep_id = "3pnayhlz"
print(sweep_id)
wandb.agent(sweep_id, function=train,project="Reporter", entity="A1_DA6401_DL")

