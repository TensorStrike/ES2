from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import wandb
import numpy as np
import math

def add_sparse_args(parser):
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='ERK', help='sparse initialization')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']




class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, args=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.name2nonzeros = {}
        self.death_rate = death_rate
        self.baseline_nonzero = None
        self.steps = 0

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix: self.prune_every_k_steps = None
        else: self.prune_every_k_steps = self.args.update_frequency

        self.cyclic_end_step = 0        # step when cyclic density ends
        self.steps_per_cycle = []
        self.current_cycle = 0
        self.cyclic_pattern = args.cyclic_pattern

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
            self.density = density
            if self.args.cyclic:
                self.density_min = density
                self.density_max = density * self.args.density_max_multiplier

            if mode == 'GMP':
                self.baseline_nonzero = 0
                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
                        self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

            elif mode == 'lottery_ticket':
                print('initialize by lottery ticket')
                self.baseline_nonzero = 0
                weight_abs = []
                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        weight_abs.append(torch.abs(weight))

                # Gather all scores in a single vector and normalise
                all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
                num_params_to_keep = int(len(all_scores) * self.density)

                threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
                acceptable_score = threshold[-1]

                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                        self.baseline_nonzero += (self.masks[name]!=0).sum().int().item()

            elif mode == 'uniform':
                self.baseline_nonzero = 0
                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda() #lsw
                        # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                        self.baseline_nonzero += weight.numel()*density

            if mode == 'ERK':
                '''
                compute a sparsity mask for each layer, distributing the overall sparsity in a way that aligns with the ERK principle
                '''
                print('initialize by ERK')
                total_params = sum(mask.numel() for mask in self.masks.values())
                expected_active_params = int(total_params * self.density)

                raw_probabilities = {}
                total_raw_prob = 0.0
                # Compute raw probabilities for each layer based on ERK principle
                for name, mask in self.masks.items():
                    n_param = mask.numel()
                    # np.prod(mask.shape) is the total # of params
                    raw_prob = (np.sum(mask.shape) / np.prod(mask.shape)) ** erk_power_scale
                    raw_probabilities[name] = raw_prob
                    total_raw_prob += raw_prob * n_param

                epsilon = expected_active_params / total_raw_prob               # scaling factor

                min_density = 0.01 * self.density                         # to prevent from layer pruned entirely

                total_nonzero = 0
                for name, mask in self.masks.items():
                    n_param = mask.numel()
                    prob_one = epsilon * raw_probabilities[name]

                    prob_one = max(prob_one, min_density)           # to ensure density is never 0
                    prob_one = min(prob_one, 1.0)                   # ensure density not above 1
                    n_ones = int(round(prob_one * n_param))         # how many parameters should be kept

                    total_nonzero += n_ones

                    # Apply pruning: keep `n_ones` parameters and set the rest to zero
                    mask_flat = mask.view(-1)
                    indices = torch.randperm(n_param, device=mask.device)[:n_ones]
                    mask_flat.zero_()
                    mask_flat[indices] = 1.0
                    self.masks[name] = mask_flat.view_as(mask)

                # print(f"Total expected active trainable params: {expected_active_params}")
                # print(f"Total actual active trainable params after ERK initialization: {total_nonzero}")

            else:
                # Handle other initialization modes
                pass

            self.apply_mask()
            self.fired_masks = copy.deepcopy(self.masks) # used for ITOP
            # self.print_nonzero_counts()

            total_size = 0
            for name, weight in self.masks.items():
                total_size  += weight.numel()
            print('Total Model parameters (trainable):', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Total parameters (trainable) under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))

    def get_cycle_position(self):
        if isinstance(self.steps_per_cycle, (int, float)):
            cycle_step = (self.steps - 1) % self.steps_per_cycle
            return cycle_step / self.steps_per_cycle
        elif isinstance(self.steps_per_cycle, list):
            cumulative_steps = 0

            for cycle_idx, cycle_length in enumerate(self.steps_per_cycle):
                cycle_start = cumulative_steps + 1
                cycle_end = cumulative_steps + cycle_length

                if cycle_start <= self.steps <= cycle_end:
                    # We're in this cycle
                    steps_into_cycle = self.steps - cycle_start
                    cycle_position = steps_into_cycle / cycle_length
                    return cycle_position

                cumulative_steps += cycle_length
            return 1.0
        else:
            return 0.0

    def calculate_cyclic_density(self, cycle_position, pattern='cosine'):

        density_range = self.density_max - self.density_min
        cycle_position = max(0.0, min(1.0, cycle_position))

        if pattern == 'cosine':
            density_factor = 0.5 * (1 - math.cos(2 * math.pi * cycle_position))
        elif pattern == 'triangular':
            if cycle_position <= 0.5:
                density_factor = 2 * cycle_position
            else:
                density_factor = 2 * (1 - cycle_position)
        else:
            raise ValueError(f"Unknown cyclic pattern: {pattern}")

        return self.density_min + density_range * density_factor


    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        self.death_rate = self.death_rate_decay.get_dr()
        self.steps += 1




        # print('step ', self.steps)
        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                if self.args.cyclic:
                    if self.steps <= self.cyclic_end_step:  # cyclic density phase
                        cycle_position = self.get_cycle_position()    # check which cycle it is currently in

                        self.next_density = self.calculate_cyclic_density(cycle_position, self.args.cyclic_pattern)

                        print('Current cycle: ', self.current_cycle)
                        # print('Cycle step:', cycle_step)
                        # print('Cycle position:', cycle_position)
                        print('Next density:', self.next_density)

                        self.prune_regrow()

                    else:
                        # for the remainder of training, reset prune_every_k_steps to value used in ITOP paper
                        self.prune_every_k_steps = 4000
                        self.truncate_weights()
                        _, _ = self.fired_masks_update()
                else:  # standard DST
                    self.truncate_weights()
                    _, _ = self.fired_masks_update()



    def add_module(self, module, density, sparse_init='ER'):
        # this initializes the mask before training
        self.modules.append(module)

        # we exclude dyrelu linear layers from parameter exploration
        for name, tensor in module.named_parameters():
            if ('relu' not in name.lower()) and ('scale' not in name):
                self.names.append(name)
                self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
                print(f"Added to masking: {name}")
            else:
                print(f"Excluded from masking: {name}")

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)

        self.init(mode=sparse_init, density=density)




    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    # reset momentum
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_prune_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density, sparse_size / total_size, epoch))

    # def prune_regrow(self):
    #     if self.next_density < self.max_cyclic_density:
    #         print(f'to grow')
    #         self.ERK_grow()
    #     else:
    #         print(f'to prune')
    #         self.ERK_prune()

    def prune_regrow(self):
        current_density = self.get_metrics()['overall_density']
        desired_density = self.next_density
        if desired_density < current_density:
            self.ERK_prune(desired_density)
        elif desired_density > current_density:
            self.ERK_grow(desired_density)

    def ERK_density_dict(self, desired_density, erk_power_scale=1.0):
        target_density = desired_density
        total_params = sum(mask.numel() for mask in self.masks.values())
        expected_active_params = int(total_params * target_density)

        raw_probabilities = {}
        total_raw_prob = 0.0

        for name, mask in self.masks.items():
            n_param = mask.numel()
            # np.prod(mask.shape) is the total number of params
            raw_prob = (np.sum(mask.shape) / np.prod(mask.shape)) ** erk_power_scale
            raw_probabilities[name] = raw_prob
            total_raw_prob += raw_prob * n_param

        epsilon = expected_active_params / total_raw_prob

        # min_density = 0.1 * self.get_metrics()['overall_density']   # to prevent layer collapse

        # compute a dict for target densities for all layers
        density_dict = {}

        for name, mask in self.masks.items():
            prob_one = epsilon * raw_probabilities[name]
            prob_one = max(prob_one, min_density)
            prob_one = min(prob_one, 1.0)
            density_dict[name] = prob_one

        return density_dict

    def ERK_prune(self, desired_density):
        '''
        prunes based on ERK principle
        '''

        # compute a dict for target densities for all layers
        density_dict = self.ERK_density_dict(desired_density)

        # prune based on the dict
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name in self.masks:
                    target_density = density_dict.get(name, 1.0)
                    n_total = self.masks[name].numel()
                    n_ones = int(target_density * n_total)
                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    mask_flat = self.masks[name].view(-1)
                    mask_flat.zero_()
                    if n_ones > 0:
                        mask_flat[idx[-n_ones:]] = 1.0
                    self.masks[name] = mask_flat.view_as(self.masks[name])

        self.apply_mask()

    def ERK_grow(self, desired_density):
        density_dict = self.ERK_density_dict(desired_density)

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name in self.masks:
                    target_density = density_dict.get(name, 1.0)
                    n_total = self.masks[name].numel()
                    n_ones = int(target_density * n_total)
                    current_n_ones = int(self.masks[name].sum().item())
                    n_to_grow = n_ones - current_n_ones     # grow this many back

                    # if n_to_grow > 0:
                    #     new_mask = self.gradient_growth(name, self.masks[name], weight)     # updated mask after apply gradient-based regrowth
                    #     self.masks[name] = new_mask.view_as(self.masks[name])           # reshape back to original shape

                    if n_to_grow > 0:
                        grad = self.get_gradient_for_weights(weight)
                        grad = grad * (self.masks[name] == 0).float()  # Consider only zeroed weights
                        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
                        n_to_grow = min(n_to_grow, (self.masks[name] == 0).sum().item())

                        mask_flat = self.masks[name].view(-1)
                        mask_flat[idx[:n_to_grow]] = 1.0
                        self.masks[name] = mask_flat.view_as(self.masks[name])

        self.apply_mask()



    def truncate_weights(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                # death
                if self.death_mode == 'magnitude':
                    new_mask = self.magnitude_death(mask, weight, name)
                elif self.death_mode == 'SET':
                    new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                elif self.death_mode == 'Taylor_FO':
                    new_mask = self.taylor_FO(mask, weight, name)
                elif self.death_mode == 'threshold':
                    new_mask = self.threshold_death(mask, weight, name)

                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.masks[name][:] = new_mask


        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth(name, new_mask, weight)

                if self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name, new_mask, weight)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask, weight)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth(name, new_mask, weight)

                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

        self.apply_mask()


    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name]==0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name]==0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def random_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask



    def momentum_neuron_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def get_metrics(self):

        total_params = 0
        active_params = 0
        normal_params = 0
        active_normal_params = 0
        mask_count = 0
        running_total = 0
        # print("Debug: Starting get_metrics")
        # print(f"Debug: Number of masks: {len(self.masks)}")


        for name, mask in self.masks.items():
            # if 'weight' in name and ('conv' in name.lower() or 'linear' in name.lower()) and 'relu' not in name.lower():
            if ('weight' in name and
             ('conv' in name.lower() or 'linear' in name.lower() or 'classifier' in name.lower() or name.endswith(
                 '.shortcut.0.weight')) and
             'bn' not in name.lower() and
             'relu' not in name.lower()):
                mask_count += 1
                param_size = mask.numel()
                layer_active = (mask != 0).sum().item()
                running_total += param_size
                # print(f"Debug: Mask {mask_count}: {name}")
                # print(f"Debug: Layer size: {param_size}, Active elements: {layer_active}")
                # print(f"Debug: Running total of parameters: {running_total}")
                normal_params += param_size
                active_normal_params += layer_active

        # Account for shared parameters
        shared_params = self.modules[0].get_shared_para()
        total_params = normal_params + shared_params

        print(f"Debug: Final normal params: {normal_params}")
        print(f"Debug: Shared params: {shared_params}")
        print(f"Debug: Total params: {total_params}")

        # active shared parameters
        # if self.args.ratio > 0:
        #     num_shared_blocks = sum(len(layer) - self.args.ratio for layer in self.modules[0].children() if
        #                             isinstance(layer, nn.ModuleList))
        #     avg_active_per_block = int(active_normal_params / (len(self.masks) / 2))  # Assuming 2 conv layers per block
        #     active_shared_params = avg_active_per_block * num_shared_blocks
        # else:
        #     active_shared_params = 0

        # print(f"Debug: Detailed breakdown of parameters:")
        # for name, param in self.modules[0].named_parameters():
        #     print(f"Layer: {name}, Params: {param.numel()}")

        active_shared_params = 0
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.modules[0], layer_name)
            if len(layer) > self.args.ratio:
                last_non_shared_block = layer[self.args.ratio - 1]
                num_shared_blocks = len(layer) - self.args.ratio
                for name, param in last_non_shared_block.named_parameters():
                    full_name = f"{layer_name}.{self.args.ratio - 1}.{name}"
                    if full_name in self.masks:
                        mask = self.masks[full_name]
                        active_params_in_block = (mask != 0).sum().item()
                        active_shared_params += active_params_in_block * num_shared_blocks

        active_params = active_normal_params + active_shared_params


        overall_density = active_params / total_params if total_params != 0 else 0
        normalized_density = active_normal_params / total_params if total_params != 0 else 0

        print(f"Debug: Normal params: {normal_params}, Active normal params: {active_normal_params}")
        print(f"Debug: Shared parameters: {shared_params}")
        print(f"Debug: Estimated active shared parameters: {active_shared_params}")
        print(f"Debug: Final counts - Total params: {total_params}, Active params: {active_params}")
        print(f"Debug: Normalized density: {normalized_density}")



        metrics = {
            'overall_density': overall_density,
            'normalized_density': normalized_density,
            'total_params': total_params,
            'active_params': active_params,
            'normal_params': normal_params,
            'active_normal_params': active_normal_params,
        }
        return metrics


    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)


        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break


    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights