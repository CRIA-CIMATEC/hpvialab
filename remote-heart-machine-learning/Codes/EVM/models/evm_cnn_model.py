import torch
from .base_model import BaseModel
from . import networks

class EVMCNNModel(BaseModel):

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.net = networks.define_evm_cnn_network(opt.init_type, opt.init_gain, self.gpu_ids)
        self.loss_names = ['Euclidian']
        self.visual_names = ['ground_truth', 'predicted'] # ['hr_value', 'predicted_hr']
        self.model_names = ['net']
        
         
        self.predicted = []
        
        if (opt.phase in ("train", "val")):
          # Apply Adam optimizer 
          self.optimizer_net = torch.optim.Adam(self.net.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
          self.optimizers.append(self.optimizer_net)
          
        self.criterionEuclidean = torch.nn.MSELoss()

    def set_input(self, input):
      """ Method to separate input 
      Parameters:
      input -- Stored parameters that will be separated to other variables
      """
      self.feature_image = input["feature_image"].to(self.device, dtype=torch.float)
      self.ground_truth = input["ground_truth"].to(self.device)
      self.image_paths = input["feature_image_paths"]
      self.subject_index = input["subject_index"]

    def forward(self):
      """Calculate predict"""
      self.predicted = self.net(self.feature_image)
    
    def compute_losses(self):
      """Method to compute loss"""
      self.loss_Euclidian = self.criterionEuclidean(self.predicted, self.ground_truth)

    def backward(self):
      """Method to compute gradient"""
      self.compute_losses()
      self.loss_Euclidian.backward()

    def optimize_parameters(self):
      """Method to record operations realized with the neural network,
      also calculate gradient norm and restart it """
      self.forward()
      self.set_requires_grad(self.net, True) # Record operations with neural network
      self.optimizer_net.zero_grad() # Restart gradient 
      self.backward()
      torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.005) # Calculate gradient norm
      self.optimizer_net.step() 