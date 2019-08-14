CFG = {
    'data_path': '../../data/Office31/Original_images/',
    'kwargs' : {'num_workers': 4},
    'batch_size': 32,
    'epoch': 200,
    'lr': 1e-3,
    'momentum': .9,
    'seed': 200,
    'log_interval': 1,
    'l2_decay': 0,
    'lambda': 10,
    'backbone': 'alexnet',
    'n_class': 31,
}