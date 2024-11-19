import argparse
import os
import yaml

def parse_args():

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--wandb_mode', default='disabled', type=str, required=False)

    # Parse the argument
    args = parser.parse_args()

    return args

def load_train_configuration(config_file):
    """
    Reads configuration for training from config file.

    :param config_file: path to config_file without extension <config_file>.yaml specifying file name of training
    configuration
    :return: dicts holding arguments for training
    """

    config_file_path = '{}.yaml'.format(config_file)
    with open(config_file_path, 'r') as file:
        file = yaml.safe_load(file)

    wandb_info = file['wandb_info']
    model_kwargs = file['model_arguments']
    dataset_kwargs = file['data_arguments']['dataset_kwargs']
    train_data_kwargs = file['data_arguments']['train_data_kwargs']
    test_data_kwargs = file['data_arguments']['test_data_kwargs']
    args = file['train_arguments']

    loss_kwargs = file['losses']

    return args, model_kwargs, dataset_kwargs, test_data_kwargs, train_data_kwargs, loss_kwargs,\
        wandb_info, config_file_path


def load_inference_configuration(config_file):
    """
    Reads configuration for inference from config file.

    :param config_file: path to config_file without extension <config_file>.yaml specifying file name of inference
    configuration
    :return: dicts holding arguments for inference
    """

    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_config', '{}.yaml'.format(config_file))
    with open(config_file_path, 'r') as file:
        file = yaml.safe_load(file)

    training_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_config', file['model_info']['train_config'])
    _, model_kwargs, dataset_kwargs, _, _, _, _, _ = load_train_configuration(training_config_file)

    # never use noise in inference
    model_kwargs['front_facing_noise'] = False

    # if dataset_kwargs are provided in inference config use them and overwrite configuration of training
    # useful if inference should be conducted on other objects than training
    if 'dataset_kwargs' in file.keys():
        dataset_kwargs.update(file['dataset_kwargs'])
    file['inference_data_kwargs'].update(dataset_kwargs)

    return file['wandb_info'], file['model_info'], file['sampling_kwargs'], model_kwargs, \
        file['inference_data_kwargs'], config_file_path