
import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms ## torchvision = librairie pytorch qui permet de faire des réseaux de neurones

import matplotlib.pyplot as plt

# check device, vérifie si le GPU est disponible
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Device = cuda ne marchera pas si je n’ai pas cuda sur mon pc, ici on utilise le cpu et non le gpu donc DEVICE = 'cpu'


#PHASE DE DEFINITION DES VARIABLES

RANDOM_SEED = 42 ## initialise l'aléa
LEARNING_RATE = 0.01 ## pas de l'algorithme
BATCH_SIZE = 16 ## définit combien d'images on va charger a chaque fois
N_EPOCHS = 30 ## définit le nombre d'époques = nombre de fois ou l’on tourne sur les données (15 fois ici)

## on fera une boucle sur les époques
# sur chque époque on fait un entrainement et on sauvegarde les train loss (valeurs de la fonction coût)

IMG_SIZE = 32 # taille des images
N_CLASSES = 10

# les classes héritent d'un module
# ici les 10 classes correspondent aux 10 types de chiffres dans lesquels on catégorise nos images
# chaque image en entrée va avoir une probabilité d'être dans une certaine classe

def train(train_loader, model, criterion, optimizer, device): # fonction d'entraînement du réseau, répétée à chaque époque
    '''
    Function for the training step of the training loop
    '''
    model.train() # => quand on lui donne des données il va essayer d’estimer les espérances/moyennes que ca pourrait donner
    
    running_loss = 0 # correspond aux valeurs de la fonction coût
    
    for X, y_true in train_loader:

        optimizer.zero_grad() # remets tous les gradients à 0
        #empeche le programme de rajouter tous les gradients au précédent
        # (sinon la fonction backward ne calculerait pas le gradient mais rajouterait les nouveaux gradients aux gradients précédents)
        
        X = X.to(device) # X en entrée
        y_true = y_true.to(device) # renvoie les valeurs du vrai vecteur
    
        # Forward pass
        y_hat, _ = model(X) # renvoie le vecteur généré à partir du modèle pour l'entrée donnée avec les poids actuels
        
        loss = criterion(y_hat, y_true) 
        # ici,compare le vecteur avec le vecteur vrai pour quantifier l’erreur que l’on fait pour des données fixées (=entropie) 
        # (que l’on multipliera par la taille de x puis divisera par la taille du dataset)
        
        
        running_loss += loss.item() * X.size(0)
        # on multiplie l'erreur par la taille de X

        # Backward pass // étape qui permet d'ajuster les poids du modèle et donc d'"apprendre"
        loss.backward() # permet de calculer automatiquement tous les gradients
        optimizer.step() # permet de faire un pas d’optimisation
        
    epoch_loss = running_loss / len(train_loader.dataset) 
    # on divise pour avoir le coût d'une époque par la taille du dataset/base de données
    
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device): ## similaire à train, mais on lance uniquement le forward du modèle

# Au lieu de train_loader, on a valid_loader =>depuis pytorch, se débrouille pour prendre des images au hasard parmi les données parcourues

    '''
    Function for the validation step of the training loop
    '''
   
    model.eval() # exploite le modèle à des fins de validations/vérifications
    running_loss = 0
    
    for X, y_true in valid_loader: # valid loader = data loader
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss



################################################## RESSOURCES GITHUB POUR CERTAINES FONCTIONS

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')


#############################################################################################################

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1): # boucle d'entrainement
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    ListeTaccuracy= []
    ListeVaccuracy= []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        

        # validation
        with torch.no_grad(): # nograd empeche le programme de calculer les gradients car inutile dans la phase de validation (pas de backward, pas de mise a jour des poids)
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device) # calcule la précision du modèle
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
            ListeTaccuracy.append(100 * train_acc)
            ListeVaccuracy.append(100 * valid_acc)

    plot_losses(train_losses, valid_losses)
    plt.figure(45)
    plt.plot(ListeTaccuracy )
    plt.plot(ListeVaccuracy)
    
    return model, optimizer, (train_losses, valid_losses)




########################################## Importation du jeu de données redimensionné

# définit une série de transformation qu’il va mettre en entrée => elles transformes en tenseurs
transforms = transforms.Compose([transforms.Resize((32, 32)), ### effectue des transformations pour redimensionner les images-sources en vecteurs/tenseurs 32x32
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data',  #  importe les images mnist
                               train=True, 
                               transform=transforms,
                               download=True) # download = true télécharge le jeu de données 

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms) # importe le dataset d'images de validation (sans entrainement)

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)

#################################################################

class LeNet5(nn.Module): # cette classe = sa manière de définir son réseau de neurones (classe qui hérite d’un module, nn= torchnetwork)

# Définit un premier module qui est une suite d’opérations (convolution tanh average pool) puis définit une deuxième suite d'opérations (out features= taille en sortie)

    def __init__(self, n_classes): # Initialise avec le nombre de classes
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(), ### Tanh correspond à la fonction d'activation cf cours opti
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) # transform x en grand tableau
        logits = self.classifier(x) # va faire passer x dans classifier puis une couche s’appelant softmax 
        probs = F.softmax(logits, dim=1)
        return logits, probs # (logits et softmax seront des vecteurs de taille 10)
    
    
# Dans forward, a une image en entrée il va sortir a probabilité d’etre dans chaque classe 
# et les logits (vecteur de prédictions bruts qu’on modèle de classification génère, valeurs brutes pas directement interprétables et converties en probas ensuite par normalisation)
    
    
    ################################# Pas encore lu
    
torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss() ## nn= torchnetwork



model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)




ROW_IMG = 10
N_ROWS = 5

fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(valid_dataset.data[index], cmap='gray_r')
    
    with torch.no_grad():
        model.eval()
        _, probs = model(valid_dataset[index][0].unsqueeze(0))
        
    title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
    
    plt.title(title, fontsize=7)
fig.suptitle('LeNet-5 - predictions');

### Quand le pas triple, learning_rate=0.003, on a de moins bons résultats d'entrainement et de validation avec plus d'oscillations et moins de convergence vers 0 pour le coût
## quand le pas diminue les valeurs de la fonction coût diminuent davantage et sont plus stables au bout de 15 époques


# le SGD devient performant sur un très grand nombre d'époques
# le SGD avec inertie donne des bons résultats et meilleurs que sans inertie uniquement lorsque la valeur du batch (nombre d'images chargées à chaque fois) est faible 
# ex: 16 à la place de 32, car l'inertie permet de se souvenir des grads précédents et donc d'avoir "artificiellement" plus de points sur lesquels s'entrainer