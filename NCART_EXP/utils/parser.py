import configargparse
import yaml


def get_parser():
    # Use parser that can read YML files
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    # parser.add('-config', '--config', required=False, is_config_file_arg=True, help='config file path',
    #            default="config/cmc.yml")  

    # parser.add('--model_name', required=True, help="Name of the model that should be trained")
    parser.add('--dataset', required=True, help="Name of the dataset that will be used")
    parser.add('--objective', required=True, type=str, default="regression", choices=["regression", "classification",
                                                                                      "binary"],
               help="Set the type of the task")
    parser.add('--use_gpu', type=bool, default=True, help="Set to true if GPU is available")
    # parser.add('--gpu_ids', type=list, default=[0], help="IDs of the GPUs used when data_parallel is true")
    parser.add('--data_parallel', type=bool, default=False, help="Distribute the training over multiple GPUs")

    parser.add('--n_trials', type=int, default=10, help="Number of trials for the hyperparameter optimization")
    parser.add('--direction', type=str, default="minimize", choices=['minimize', 'maximize'],
               help="Direction of optimization.")

    parser.add('--num_splits', type=int, default=5, help="Number of splits done for cross validation")
    parser.add('--shuffle', type=bool, default=True, help="Shuffle data during cross-validation")
    parser.add('--seed', type=int, default=2023, help="Seed for KFold initialization.")
    parser.add('--scale', type=bool, default=True, help="Normalize input data.")
    parser.add('--target_encode', action="store_true", help="Encode the targets that they start at 0. (0, 1, 2,...)")
    parser.add('--one_hot_encode', action="store_true", help="OneHotEncode the categorical features")

    parser.add('--batch_size', type=int, default=1024, help="Batch size used for training")
    parser.add('--val_batch_size', type=int, default=256, help="Batch size used for training and testing")
    parser.add('--early_stopping_rounds', type=int, default=10, help="Number of rounds before early stopping applies.")
    parser.add('--epochs', type=int, default=1000, help="Max number of epochs to train.")
    parser.add('--logging_period', type=int, default=100, help="Number of iteration after which validation is printed.")

    parser.add('--num_features', type=int, required=True, help="Set the total number of features.")
    parser.add('--num_classes', type=int, default=1, help="Set the number of classes in a classification task.")
    parser.add('--cat_idx', type=int, action="append", help="Indices of the categorical features")
    parser.add('--cat_dims', type=int, action="append", help="Cardinality of the categorical features (is set "
                                                             "automatically, when the load_data function is used.")


    return parser


