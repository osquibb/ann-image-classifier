import argparse
from torch import nn
from torch import optim
import utilities as util
import model as mod
from os import listdir, mkdir

def main():
    in_arg = get_input_args() # Creates and returns command line arguments
    
    print('\nData Directory:\n', in_arg.data_directory, '\n')
    
    print('Optional Command Line Arguments:\n', 'Save Checkpoint [--save_dir]: ', in_arg.save_dir, '\n', 'Pretrained Network [--arch]: ', in_arg.arch, '\n', 'Learning Rate [--learning_rate]: ', in_arg.learning_rate,'\n' , 'Hidden Units [--hidden_units]: ', in_arg.hidden_units,'\n' , 'Epochs [--epochs]: ', in_arg.epochs,'\n' , 'GPU [--gpu]: ', in_arg.gpu, '\n')
    
    if 'checkpoints' not in listdir(): # makes checkpoints folder if it doesn't already exist
        mkdir('checkpoints')
    
    train_dir, valid_dir, test_dir = util.get_data(in_arg.data_directory) # Returns Train, Validation and Test Directories
    
    transformed_train, transformed_valid, transformed_test = mod.transform_data(train_dir, valid_dir, test_dir) # Returns transformed datasets
    
    train_loader, valid_loader, test_loader = mod.load_data(transformed_train, transformed_valid, transformed_test) # Returns Data loaders
    
    model = mod.build_model(util.label_count(train_dir), in_arg.hidden_units, in_arg.arch, transformed_train.class_to_idx) # Returns built model
    
    epochs = in_arg.epochs # Epochs initially set by command line argument in_arg.epochs.  Can be changed with m.load_checkpoint()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    
    use_gpu = mod.use_gpu(model, in_arg.gpu) # Returns True or False for GPU use
    
    mod.train(model, criterion, optimizer, train_loader, valid_loader, use_gpu, in_arg.epochs) # Trains the model.  Prints Training Loss, Validation Loss & Validation Accuracy
    
    mod.save_checkpoint(in_arg.arch, model.classifier.state_dict(), transformed_train.class_to_idx, util.label_count(train_dir), in_arg.hidden_units, in_arg.epochs, in_arg.save_dir) # Saves classifier and other model parameters to checkpoint
    
def get_input_args():
    parser = argparse.ArgumentParser() # Creates the command line argument parser
    parser.add_argument('data_directory', type=str) # Required Argument
    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                        help='Set directory to save checkpoints') # Optional Argument
    parser.add_argument('--arch', type=str, default='vgg11', 
                        help='Choose architecture (eg: vgg11)') # Optional Argument
    parser.add_argument('--learning_rate', type=float, default=0.002, 
                        help='Set learning rate (eg: 0.01)') # Optional Argument
    parser.add_argument('--hidden_units', type=int, default=1024, 
                        help='Set hidden units (eg: 512)') # Optional Argument
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Set epochs (eg: 3)') # Optional Argument
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU (True or False)') # Optional Argument
    return parser.parse_args() # Returns collection of parsed arguments



if __name__ == "__main__":
    main()
    
    