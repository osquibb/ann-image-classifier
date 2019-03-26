import argparse
from torch import nn
from torch import optim
import model as mod
import utilities as util

def main():
    in_arg = get_input_args() # Creates and returns command line arguments

    print('\nPath To Image:\n', in_arg.path_to_image, '\n', '\nCheckpoint:\n', in_arg.checkpoint, '\n')
    
    print('Optional Command Line Arguments:\n', 'Top K [--top_k]: ', in_arg.top_k, '\n', 'Category Names [--category_names]: ', in_arg.category_names, '\n', 'GPU [--gpu]: ', in_arg.gpu, '\n')

    label_count, hidden_units, arch, class_to_idx, classifier_state_dict, epochs = mod.load_checkpoint(in_arg.checkpoint, in_arg.gpu) # Load checkpoint
        
    model = mod.build_model(label_count, hidden_units, arch, class_to_idx) # Build model
    
    model.classifier.load_state_dict(classifier_state_dict)
    criterion = nn.NLLLoss()
    
    image = util.process_image(in_arg.path_to_image) # Pre-process image
    
    labels = util.get_labels(in_arg.category_names) # Get dict of categories mapped to real names
    
    mod.predict(image, model, labels, in_arg.top_k, in_arg.gpu) # Prints Top K Labels and Probabilities  

def get_input_args():
    parser = argparse.ArgumentParser() # Creates the command line argument parser
    parser.add_argument('path_to_image', type=str) # Required Argument
    parser.add_argument('checkpoint', type=str) # Required Argument
    parser.add_argument('--top_k', type=int, default=1,
                        help='Set K for Top K most likely classes') # Optional Argument
    parser.add_argument('--category_names', type=str, default = None, 
                        help='Filepath of JSON object for mapping of categories to real names') # Optional Argument
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU (True or False)') # Optional Argument
    return parser.parse_args() # Returns collection of parsed arguments



if __name__ == "__main__":
    main()