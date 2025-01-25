import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
# from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, extract, exists, default
from tqdm.autonotebook import tqdm


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        # self.logger("Total Number of parameter: %.2fM" % (total_num / 1e6))
        # self.logger("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                            mode='min',
                                                            verbose=True,
                                                            threshold_mode='rel',
                                                            **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            model_states['ori_latent_mean'] = self.net.ori_latent_mean
            model_states['ori_latent_std'] = self.net.ori_latent_std
            model_states['cond_latent_mean'] = self.net.cond_latent_mean
            model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        #self.logger(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        #self.logger(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        # self.logger(self.net.ori_latent_mean)
        # self.logger(self.net.ori_latent_std)
        # self.logger(self.net.cond_latent_mean)
        # self.logger(self.net.cond_latent_std)

    def predict_x_high(self, x_t, t, x_low, T=1000): 
        noise = torch.randn_like(x_t)
        # m_t = t/T 
        # var_t = 2*(m_t - m_t*m_t)
        m_t = torch.tensor(t/T, device=x_low.device, dtype=x_low.dtype)
        var_t = torch.tensor(2*(m_t - m_t*m_t), device=x_low.device, dtype=x_low.dtype)
        # m_t = extract(m_t, t, x_t.shape)
        # var_t = extract(var_t, t, x_t.shape)
        sigma_t = torch.sqrt(var_t)
        x_high= (x_t - m_t * x_low - sigma_t * noise) / (1. - m_t)
        return x_high 
        

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
       
        (x_t, y_high), (x_low, y_low) = batch

        x_high =  self.predict_x_high(x_t,self.config.t_of_high, x_low)
        
        torch.manual_seed(step)
        rand_mask = torch.rand(y_high.size())
        mask = (rand_mask <= self.config.training.classifier_free_guidance_prob)
        
        # mask y_high and y_low
        y_high[mask] = 0.
        y_low[mask] = 0.
            
        x_high = x_high.to(self.config.training.device[0])
        y_high = y_high.to(self.config.training.device[0])
        x_low = x_low.to(self.config.training.device[0])
        y_low = y_low.to(self.config.training.device[0])

        loss, additional_info = net(x_high, y_high, x_low, y_low)
        # if write:
        #     self.writer.add_scalar(f'loss/{stage}', loss, step)
        #     if additional_info.__contains__('recloss_noise'):
        #         self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
        #     if additional_info.__contains__('recloss_xy'):
        #         self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, low_candidates, low_scores, high_cond_scores):
        low_candidates = low_candidates.to(self.config.training.device[0])
        low_scores = low_scores.to(self.config.training.device[0])
        high_cond_scores = high_cond_scores.to(self.config.training.device[0])
        high_candidates = net.sample(low_candidates, low_scores, high_cond_scores, clip_denoised=self.config.testing.clip_denoised, classifier_free_guidance_weight=self.config.testing.classifier_free_guidance_weight)
        
        return high_candidates