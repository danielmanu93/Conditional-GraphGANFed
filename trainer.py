import numpy as np
import os
import time
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import utils
# from utils import *
from Gen_Discr_models import Generator, Discriminator, Critic
from molecular_dataset import *
from molecular_dataset import MolecularDataset
import matplotlib.pyplot as plt
# from utils import classification_report, mols2grid_image, reconstructions

class Trainer(object):

    def __init__(self, args, data, idxs):
        """Initialize configurations."""

        self.args = args

        # Data loader.
        self.data = MolecularDataset()
        self.data.load(args.mol_data_dir)

        # Model configurations.
        self.z_dim = args.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.c_conv_dim = args.c_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.d_repeat_num = args.d_repeat_num
        self.lambda_cls = args.lambda_cls
        self.lambda_rec = args.lambda_rec
        self.lambda_gp = args.lambda_gp
        self.post_method = args.post_method

        # Training configurations.
        self.batch_size = args.batch_size
        self.num_iters_local = args.num_iters_local
        self.num_iters_decay = args.num_iters_decay
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.c_lr = args.c_lr
        self.dropout = args.dropout
        self.n_critic = args.n_critic
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.resume_iters = args.resume_iters
        self.epochs_global = args.epochs_global
        self.frac = args.frac
        self.num_users = args.num_users
        self.metric = "all"  # ["all", "groupdep"]

        # Test configurations.
        self.test_iters = args.test_iters

        # Miscellaneous.
        self.use_tensorboard = args.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.model_save_dir = args.model_save_dir
        self.result_dir = args.result_dir

        # Step size.
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step
        self.lr_update_step = args.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
            
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.z_dim,
                           self.data.vertexes,
                           self.data.bond_num_types,
                           self.data.atom_num_types,
                           self.dropout)
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)
        self.C = Critic(self.c_conv_dim, self.m_dim, self.b_dim, self.dropout)

        #global model with both generator & discriminator
        # self.g_global_model = self.G
        # self.d_global_model = self.D

        self.g_optimizer = torch.optim.Adam(list(self.G.parameters())+list(self.C.parameters()),
                                            self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.c_lr, [self.beta1, self.beta2])
        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)
        self.C.to(self.device)

        return self.G, self.D, self.C
    
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()  #get no. of params iteratively
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, c_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = c_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size())+[dim]).to(self.device)
        out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)
        return out

    def sample_z(self, batch_size):   #draw random samples from normal Gaussian distr.
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]
    

    def reward(self, mols):
        rr = 0.5
        reward = 0
        
        for score in ('logp,qed,sim,div,valid,unique' if self.metric == 'all' else 'logp,qed,sim,valid,unique' if self.metric=='groupdep' else self.metric).split(','):
            
            # for group dependence, different dependence options can be set
            # for example if optimizing logp such that a specific threshold for validity is met

            if score == 'sim':
                sim = rr * utils.MolecularMetrics.similarity_scores(mols, self.data)
                reward -= sim
            elif score == 'logp':
                logp = rr * utils.MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
                reward += logp
            elif score == 'qed':
                qed = utils.MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
                reward += qed
            elif score == 'novel':
                novel = rr * utils.MolecularMetrics.novel_scores(mols, self.data)
                reward += novel
            elif score == 'unique':
                unique = rr * utils.MolecularMetrics.unique_scores(mols)
                reward += unique
            elif score == 'div':
                div = rr * utils.MolecularMetrics.diversity_scores(mols, self.data)
                reward += div
            elif score == 'valid':
                valid = rr * utils.MolecularMetrics.valid_scores(mols)
                reward += valid
            else:
                raise RuntimeError('{} is not defined as a metric'.format(score))
            
        #     if score == 'sim':
        #         sim = utils.similarity_scores(mols, data)
        #         rewards.append(sim)
        #     elif score == 'logp':
        #         logp = utils.water_octanol_partition_coefficient_scores(mols, norm=True)
        #         rewards.append(logp)
        #     elif score == 'qed':
        #         qed = utils.quantitative_estimation_druglikeness_scores(mols, norm=True)
        #         rewards.append(qed)
        #     elif score == 'unique':
        #         unique = utils.unique_scores(mols)
        #         rewards.append(unique)
        #     elif score == 'valid':
        #         valid = utils.valid_scores(mols)
        #         rewards.append(valid)

        # return torch.cat(rewards, dim=1).reshape(-1, len(rewards))

        # print("reward", reward)
        #return rr.reshape(-1, 1)
        return reward

    def tnr(self, model, global_round, client_id):

        global z
        global edges_hard, nodes_hard
        global edges_hat, nodes_hat

        # thresh_val = 0.8  # Set your desired threshold metric value
        # thresh_metric = 1  # Set the metric index to be optimized (e.g., logP: 0, QED: 1, etc.)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        c_lr = self.c_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        gen_loss = []
        dis_loss = []
        crit_loss = []
        first_crit_div_loss = []
        last_crit_div_loss = []
        # crit_loss = []
        # g_batch_loss = []
        # d_batch_loss = []
        # c_batch_loss = []
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters_local):
            if (i+1) % self.log_step == 0:
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                z = self.sample_z(a.shape[0])
                print('[Valid]', '')
            else:
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                z = self.sample_z(self.batch_size)
                

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
            x = torch.from_numpy(x).to(self.device).long()            # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)
            z = torch.from_numpy(z).to(self.device).float()
            
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            
            # Compute loss with real images.
            logits_real, features_real = self.D(a_tensor, None, x_tensor)
            d_loss_real = - torch.mean(logits_real)

            # Compute loss with fake images.
            edges_logits, nodes_logits = self.G(z)
            
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            d_loss_fake = torch.mean(logits_fake)

            # Compute loss for gradient penalty.
            eps = torch.rand(logits_real.size(0),1,1,1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, None, x_int1)
            d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

            # Backward and optimize.
            d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()


            # =================================================================================== #
            #                  3. Train the Critic                                                #
            # =================================================================================== #
            # Z-to-target
            edges_logits, nodes_logits = self.G(z)

            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), "soft_gumbel")

            # Real Reward
            rewardR = torch.from_numpy(self.reward(mols)).to(self.device)

            # Fake reward
            (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                    for e_, n_ in zip(edges_hard, nodes_hard)]
            rewardF = torch.from_numpy(self.reward(mols)).to(self.device)

            # RL loss (MSE error)
            value_logit_real,_ = self.C(a_tensor, None, x_tensor, torch.sigmoid)
            value_logit_fake,_ = self.C(edges_hat, None, nodes_hat, torch.sigmoid)
            # print("fake:", value_logit_fake)
            
            # fake_reward_loss = (value_logit_fake - rewardF) ** 2
            # real_reward_loss = (value_logit_real - rewardR) ** 2

            mse_loss = torch.mean((value_logit_real - rewardR) ** 2 + (value_logit_fake - rewardF) ** 2)

            # Apply the constraint check
            # constraint_satisfied = (torch.mean(rewardF[:, thresh_metric]) > thresh_val)

            # Update the critic based on the reward loss if the constraint is satisfied
            # if constraint_satisfied:
            #     c_loss = torch.mean(fake_reward_loss + real_reward_loss)

                # Backward and optimize for critic
                # self.reset_grad()
                # c_loss.backward()
                # self.c_optimizer.step()

            # Backward and optimize for critic
            c_loss = mse_loss
            self.reset_grad()
            c_loss.backward()
            self.c_optimizer.step()

            # =================================================================================== #
            #                  3. Train the generator                                             #
            # =================================================================================== #
            
            detached_value_logit_fake = value_logit_fake.detach()
            # print("detached_fake:", detached_value_logit_fake)

            # crit_loss.append(c_loss.item())
            # print("Critic loss: ", crit_loss) 

            if (i+1) % self.n_critic == 0:
                
                # Z-to-target
                edges_logits, nodes_logits = self.G(z)

                # Postprocess with Gumbel softmax
                (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
                logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
                g_loss_fake = - torch.mean(logits_fake)

                # Backward and optimize for generator.
                # g_loss = g_loss_fake + c_loss
                g_loss = (g_loss_fake - detached_value_logit_fake).mean()  # fake_reward_loss
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss'] = g_loss.item()
                loss['Critic/loss'] = c_loss.item()

                gen_loss.append(g_loss.item())

            dis_loss.append(d_loss.item())  
            crit_loss.append(c_loss.item())
            first_crit_div_loss.append(c_loss.item())
            last_crit_div_loss.append(c_loss.item())

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(timedelta(seconds=et))[:-7]
                log = "Global Round [{}], Elapsed [{}], Iteration [{}/{}]".format(global_round, et, i+1, self.num_iters_local)

                # Log update
                m0, m1 = utils.all_scores(mols, self.data, sample=True, norm=True)     #'mols' is output of Fake Reward
                m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                m0.update(m1)
                loss.update(m0)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
   
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                C_path = os.path.join(self.model_save_dir, '{}-C.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.C.state_dict(), C_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters_local - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                c_lr -= (self.c_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr, c_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}, c_lr: {}.'.format(g_lr, d_lr, c_lr))

        # d_epoch_loss.append(sum(d_batch_loss)/len(d_batch_loss))
        # g_epoch_loss.append(sum(g_batch_loss)/len(g_batch_loss))
        # c_epoch_loss.append(sum(c_batch_loss)/len(c_batch_loss))
        # print(" "+"\n")
        # print("------------- Model local losses -----------"+"\n")
        # print("Disc losses", dis_loss)
        # print("Gen losses", gen_loss)
        # print("Crit losses", crit_loss)

        for i in range(self.num_iters_local - 990, self.num_iters_local):
                first_crit_div_loss[i] /= 10.0
        
        for i in range(self.num_iters_local):
                last_crit_div_loss[i] /= 100.0

        return self.G.state_dict(), self.D.state_dict(), self.C.state_dict(), gen_loss, dis_loss, crit_loss, first_crit_div_loss, last_crit_div_loss

    def test(self):
        # Load the trained generator.
        global edges_hard, nodes_hard
        global edges_hat, nodes_hat
        self.restore_model(self.test_iters)

        with torch.no_grad():
            mols, _, _, a, x, _, _, _, _ = self.data.next_test_batch()
            z = self.sample_z(a.shape[0])
            z = torch.from_numpy(z)

            # Z-to-target
            edges_logits, nodes_logits = self.G(z.float())
            
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            g_loss_fake = - torch.mean(logits_fake)

            # Preprocess with hard Gumbel
            (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                    for e_, n_ in zip(edges_hard, nodes_hard)]

            # Print out testing information.
            start_time = time.time()
            start_iters = self.test_iters
            num_iters_local = 1010
            for i in range(start_iters, num_iters_local):
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, num_iters_local)
                    # Log update
                    m0, m1 = utils.all_scores(mols, self.data, sample=True, norm=True)     # 'mols' is output of Fake Reward
                    m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                    m0.update(m1)
                    
                    for tag, value in m0.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log) 

            recons_mols = utils.reconstructed_mols(self.data, sample=True)
            
            #print the reconstructed image
            recons_image = utils.mols2grid_image(recons_mols[:30], molsPerRow = 5)
            recons_image.save("/vast/home/dmanu/Desktop/fedgan3/mols_img/recons_image.png", dpi=(1000, 1000))
            recons_image.show()

            #print generated image
            # img = utils.mols2grid_image(mols[:5], molsPerRow = 5) 
            # img.show()
            # img.save("/Users/daniel/Desktop/PhD materials/Fed-GNN-GAN/fedgan/mols_img/mols_grid.png")

