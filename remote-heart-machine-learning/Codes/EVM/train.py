import time
try:
    from .options.train_options import TrainOptions
    from .data import create_dataset
    from .models import create_model
    from .util.visualizer import Visualizer
except:
    from options.train_options import TrainOptions
    from data import create_dataset
    from models import create_model
    from util.visualizer import Visualizer
from collections import OrderedDict

def train(opt):
    train_dataset, val_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    val_total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        train_epoch_losses = OrderedDict({name : [] for name in model.loss_names})
        val_epoch_losses = OrderedDict({name : [] for name in model.loss_names})
        
        print("Train Phase")
        
        for i, data in enumerate(train_dataset, start = 1):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
            
            result = model.get_current_visuals()
            losses = model.get_current_losses()
            for key in losses.keys():
                train_epoch_losses[key].append(losses[key]) # OrderedDict({key : train_epoch_losses[key] + losses[key] for key in losses.keys()})
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
              
              t_comp = (time.time() - iter_start_time) / opt.batch_size
              visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
              
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        train_epoch_losses = OrderedDict({key : sum(train_epoch_losses[key]) / len(train_epoch_losses[key]) for key in train_epoch_losses.keys()})
        print(f"train_epoch_losses: {train_epoch_losses}")
        visualizer.get_losses(epoch, float(epoch_iter) / dataset_size, train_epoch_losses)
        
        epoch_iter = 0
        iter_data_time = time.time()
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        print("Validation Phase")
        
        for i, data in enumerate(val_dataset, start = 1):
            iter_start_time = time.time()  # timer for computation per iteration
            val_total_iters += opt.batch_size
            model.set_input(data)
            model.evaluate()
            epoch_iter += opt.batch_size
            losses = model.get_current_losses()
            result = model.get_current_visuals()
            
            for key in losses.keys():
                val_epoch_losses[key].append(losses[key])   # Store loss 
                
            if val_total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time # Calculate time 
                t_comp = (time.time() - iter_start_time) / opt.batch_size # Calculate time per data
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

        val_epoch_losses = OrderedDict({key : sum(val_epoch_losses[key]) / len(val_epoch_losses[key]) for key in val_epoch_losses.keys()}) # Calculate  validation loss per epoch
        visualizer.get_val_losses(val_epoch_losses)
        print(f"val_epoch_losses: {val_epoch_losses}")

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    visualizer.plot_losses()

if __name__ == '__main__':
    opt = TrainOptions().parse() # get train options

    train(opt)