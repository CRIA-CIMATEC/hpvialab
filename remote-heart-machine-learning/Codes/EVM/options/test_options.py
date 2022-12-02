from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--test', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--model_suffix', type=str, default='net', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        # parser.add_argument('--num_test', type=int, default=50, help='how many test images to run') # USE max_dataset_size to control how many test images you want.
        # rewrite devalue values
        parser.set_defaults(dataset_mode='single')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
