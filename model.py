import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image

def transform_data(train_dir, valid_dir, test_dir):
    '''Input train directory, validation directory and test directory. Returns transformed datasets for each'''
    
    data_transforms = {'train': transforms.Compose([transforms.RandomHorizontalFlip(p=0.1),
                                                    transforms.RandomVerticalFlip(p=0.1),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])
                      }

    transformed_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                            'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['test']),
                            'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
                           }  # Validation set will use testing_transforms

    return transformed_datasets['train'], transformed_datasets['valid'], transformed_datasets['test']

def load_data(transformed_train, transformed_valid, transformed_test):
    '''Input transformed datasets (train, valid, test).  Returns dataloaders for each'''
    
    dataloaders = {'train': torch.utils.data.DataLoader(transformed_train, batch_size=64, shuffle=True),
                   'validation': torch.utils.data.DataLoader(transformed_valid, batch_size=32),
                   'test': torch.utils.data.DataLoader(transformed_test, batch_size=32)
                  }
    
    return dataloaders['train'], dataloaders['validation'], dataloaders['test']

def build_model(label_count, hidden_units, arch, class_to_idx, drop_p=0.1):
    '''Input Pretrained Network name (vgg11, alexnet or densenet161) (default='vgg11'), labels dictionary, number of hidden units, class_to_idx and drop probability (default=0.1).  Returns the model'''
    
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        classifier_in_features = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        classifier_in_features = 9216
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        classifier_in_features = 2208
    else:
        print('Pretrained Network Not Available')
    
    for parameters in model.parameters():
        parameters.requires_grad = False
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_in_features, hidden_units)),
        ('ReLU', nn.ReLU()),
        ('Dropout', nn.Dropout(drop_p)),
        ('fc2', nn.Linear(hidden_units, label_count)),
        ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    
    return model

def use_gpu(model, gpu):
    '''Returns True if GPU is available and runs model.cuda().  Returns False and model.cpu() if not.'''
    
    if gpu:
        model.to('cuda')
    else:
        model.to('cpu')
        
    return gpu
    
def train(model, criterion, optimizer, train_loader, valid_loader, use_gpu, epochs):
    '''Trains the model.  Prints Training Loss, Validation Loss & Validation Accuracy.'''
        
    model.train()
    print_every = 40
    steps = 0
    
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in iter(train_loader):
            steps += 1
            if use_gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            
            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, criterion, valid_loader, use_gpu)
                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(validation_loss),
                      "Validation Accuracy: {:.3f}".format(accuracy))             
                
def validate(model, criterion, valid_loader, use_gpu):
    '''Input Model, Criterion, Valid_loader.  Returns Validation Loss & Accuracy '''
    
    model.eval()
    accuracy = 0
    validation_loss = 0
    
    for images, labels in iter(valid_loader):
        if use_gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            images, labels = Variable(images), Variable(labels)
        
        output = model.forward(images)
        validation_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        
        if use_gpu:
            accuracy += equality.type(torch.cuda.FloatTensor).mean()
        else:
            accuracy += equality.type(torch.DoubleTensor).mean()
        
    return validation_loss/len(valid_loader), accuracy/len(valid_loader)
    
def save_checkpoint(arch, classifier_state_dict, class_to_idx, label_count, hidden_units, epochs, checkpoint_save_path):
    '''Saves model to checkpoint_save_path + checkpoint.pth.'''
    
    checkpoint = {'arch': arch,
                  'classifier_state_dict': classifier_state_dict,
                  'class_to_idx': class_to_idx,
                  'label_count': label_count,
                  'hidden_units': hidden_units,
                  'epochs': epochs
                 }
    
    checkpoint_filepath = checkpoint_save_path + 'checkpoint.pth'
    torch.save(checkpoint, checkpoint_filepath)
    
def load_checkpoint(checkpoint_filepath, use_gpu):
    '''Loads model from checkpoint'''
    
    if use_gpu:
        checkpoint = torch.load(checkpoint_filepath)
    else:
        checkpoint = torch.load(checkpoint_filepath, map_location=lambda storage, loc: storage)
    arch = checkpoint['arch']
    classifier_state_dict = checkpoint['classifier_state_dict']
    class_to_idx = checkpoint['class_to_idx']
    label_count = checkpoint['label_count']
    hidden_units = checkpoint['hidden_units']
    epochs = checkpoint['epochs']
    
    return label_count, hidden_units, arch, class_to_idx, classifier_state_dict, epochs
    
def predict(preprocessed_image, model, label_names, topk, gpu):
    ''' Predicts the class (or classes) of an image using a trained model.  Prints Top K Labels and Probabilities'''
    
    use_gpu(model, gpu)
    model.eval()
    tensor = torch.from_numpy(preprocessed_image)
    
    if gpu:
        inputs = Variable(tensor.float().cuda())

    else:
        model.type(torch.DoubleTensor)
        inputs = Variable(tensor)
        
    inputs.unsqueeze_(0)
    output = model.forward(inputs)
    ps = torch.exp(output).data.topk(topk)
    
    if use_gpu:
        probs_tensor = ps[0].cpu()
    else:
        probs_tensor = ps[0]
        
    if use_gpu:
        classes_tensor = ps[1].cpu()
    else:
        classes_tensor = ps[1]
    
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    topk_classes = []
    probs = []
    
    classes_numpy = classes_tensor.numpy()
    probs_numpy = probs_tensor.numpy()
    
    for i in classes_numpy[0]:
        topk_classes.append(class_to_idx_inverted[i])
        
    for i in probs_numpy[0]:
        probs.append(i)
        
    labels = []
    
    if label_names == None:
        labels = topk_classes
    else:
        for cl in topk_classes:
            labels.append(label_names[cl])
        
    for label, prob in zip(labels, probs):
        print(' Label: ', label, '\n', 'Probability: ', prob, '\n')
                  
            
    
        
            
            
        
                
                
    
    
    
    
    
    
                                                                  
    


    