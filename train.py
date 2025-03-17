### DA6401 - ASSIGNMENT - 01 (AE21B105)  ###
# Importing the necessary libraries #
import numpy as np
import source
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import fashion_mnist, mnist
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Argument parser #
parser = argparse.ArgumentParser(description="Training a neural network with backpropagation !!!")
# adding the arguments #
parser.add_argument("-wp", '--wandb_project', type=str, default="projectname", help = "project name used in wandb dashboard to track experiments")
parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb enetity used to track the experiments")
parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help = "Dataset on which the model will be trained")
parser.add_argument("-e", "--epochs", type=int, default=5, help = "Number of epochs to train the model")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size used to train the network")
parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function to use")
parser.add_argument("-o", "--optimizer", type=str, default="adam", choices=["sgd","momentum","nag","rmsprop","adam","nadam"], help="optimizer to be used in training")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate used in the gradient update")
parser.add_argument("-m", "--momentum", type=float, default=0.98, help="Momentum used in momentum-GD and NAG")
parser.add_argument("-beta", "--beta", type=float, default=0.95, help="Beta used by RMSprop")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta_1 used by Adam and NAdam")
parser.add_argument("-beta2", "--beta2",type=float, default=0.99, help="Beta_2 used by Adan and NAdam")
parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon used by the optimizers")
parser.add_argument("-w_d", "--weight_decay", type=float, default=5e-4, help="Weight decay used in training (L2)")
parser.add_argument("-w_i", "--weight_init", type=str, default="Xavier", choices=["Xavier", "random"], help = "Weight initialization for the params")
parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="number of hidden layers")
parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of neurons per hidden layer")
parser.add_argument("-a", "--activation", type=str, default="ReLu", choices=["identity","sigmoid","tanh","ReLu"], help = "activation to be used for the hidden layers")
# parsing the arguments
args = parser.parse_args()

# WandB limport and login
import wandb
wandb.login()

# Importing and reshaping dataset
if args.dataset == "fashion_mnist":
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # class names if fashion_mnist
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
else:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # class names if mnist
    class_names = ["0","1","2","3","4","5","6","7","8","9"]

# Splitting the dataset into train and validation .. test data remains untouched
X_train,X_cv,y_train,y_cv = train_test_split(X_train,y_train,test_size=0.1,stratify=y_train)
# Reshaping the dataset to feed into neural network (Note the inputs are normalized and output are converted to one hot vectors)
X_train = X_train.reshape(54000,-1)/255.0
X_cv = X_cv.reshape(6000,-1)/255.0
X_test = X_test.reshape(10000,-1)/255.0
y_train = source.one_hot_numpy(y_train)
y_cv = source.one_hot_numpy(y_cv)
y_test = source.one_hot_numpy(y_test)

config = {
             "optimizer" : args.optimizer,
             "loss" : args.loss,
             "dataset" : args.dataset,
             "epochs" : args.epochs,
             "batch_size" : args.batch_size,
             "learning_rate" : args.learning_rate,
             "weight_decay" : args.weight_decay,
             "weight_init" : args.weight_init,
             "num_layers" : args.num_layers,
             "hidden_size" : args.hidden_size,
             "activation" : args.activation
}
# initiate a wandb run
run = wandb.init(entity = args.wandb_entity, project=args.wandb_project, config=config)

# Initiate the model with the parsed arguments
# Architecture definition
arch = [784]
activation_sequence = []
for i in range(1,args.num_layers+1):
    neurons = args.hidden_size
    activation = args.activation
    arch.append(neurons)
    activation_sequence.append(activation)
# finalize 
arch.append(10)
activation_sequence.append("softmax")
# printing the architecture and activation sequence
print(arch, activation_sequence)
# Other hyper-parameters of the model
optimizer = args.optimizer
learning_rate = args.learning_rate
loss = args.loss
initialization = args.weight_init
momentum = args.momentum
weight_decay = args.weight_decay
beta_rms = args.beta
beta_1 = args.beta1
beta_2 = args.beta2
epsilon = args.epsilon
# Model definition
md1 = source.FeedForwardNeuralNetwork(arch=arch, activation_sequence=activation_sequence, optimizer=optimizer,
                                      learning_rate=learning_rate,eps=epsilon,weight_decay= weight_decay, loss=loss,initialization=initialization,momentum=momentum,beta_rms=beta_rms,
                                      beta_1=beta_1,beta_2=beta_2)

# Batchloader for the model training
batch_train = source.Batchloader(X_train, y_train, batch_size=args.batch_size,shuffle=True)

# Model training ...
epochs = args.epochs
print("Model training is starting")
for epoch in range(1,epochs+1):
    for X_, y_ in batch_train:
        md1.train_step(X_, y_, epoch)
    train_pred = md1.forward_call(inputs_=X_train)
    test_pred = md1.forward_call(inputs_=X_cv)

    loss_train = source.find_loss(y_pred= train_pred, y_true =y_train, loss=args.loss)
    loss_cv = source.find_loss(y_pred= test_pred, y_true = y_cv, loss = args.loss)

    train_pred = md1.forward_call(inputs_=X_train,threshold=True)
    test_pred = md1.forward_call(inputs_=X_cv,threshold=True)

    accuracy_train = source.accuracy(train_pred, y_train)
    accuracy_cv = source.accuracy(test_pred,  y_cv)

    run.log(data={"loss_train" : loss_train, "loss_cv" : loss_cv, "acc_train" : accuracy_train, "acc_cv" : accuracy_cv})
    if epoch%1 == 0:
        print(F"Epoch {epoch} || loss_train : {loss_train} | loss_test : {loss_cv} || acc_train : {accuracy_train} | acc_test : {accuracy_cv} ||")

print("Model training is done ...")

# prediction of the test data
test_pred = md1.forward_call(inputs_=X_test,threshold=True)
run.log(data = {"test_acc" : source.accuracy(test_pred,y_test)})
from sklearn.metrics import confusion_matrix

test_pred_ = np.argmax(test_pred, axis=1)
test_true = np.argmax(y_test, axis = 1)
cm = confusion_matrix(test_true, test_pred_)
wandb.log({
    "Confusion Matrix": wandb.plot.confusion_matrix(
        y_true=test_true,
        preds=test_pred_,
        class_names=class_names,
        title="Confusion Matrix"
    )
})

# Plotting confusion matrix
plt.figure(figsize=(9,7))
font = {'family' : 'Comic Sans MS','weight' : 'normal', 'size'   : 12}
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names,cbar=False)
plt.title("Confusion Matrix", fontdict= font)
plt.xlabel("Predicted",fontdict=font)
plt.ylabel("True", fontdict=font)
plt.tight_layout()
# Logging the plot to wandb
wandb.log({"Confusion Matrix_chk": wandb.Image(plt.gcf())})
plt.close()

wandb.finish()
