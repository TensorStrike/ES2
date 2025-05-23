from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from sparselearning.dyrelu import DyReLUB, DyReLUB_inf
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, \
    ResNet50, ResNet18, ResNet50_ImageNet

from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders, get_imagenet_dataloaders
import torchvision
import torchvision.transforms as transforms
import warnings
import wandb
from fvcore.nn import FlopCountAnalysis



warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['MLPCIFAR10'] = (MLP_CIFAR10, [])
models['lenet5'] = (LeNet_5_Caffe, [])
models['lenet300-100'] = (LeNet_300_100, [])
models['ResNet34'] = ()
models['ResNet18'] = ()
models['alexnet-s'] = (AlexNet, ['s', 10])
models['alexnet-b'] = (AlexNet, ['b', 10])
models['vgg-c'] = (VGG16, ['C', 10])
models['vgg-d'] = (VGG16, ['D', 10])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-2'] = (WideResNet, [28, 2, 10, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])
models['ResNet50'] = ()



def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density,
                                               hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def calculate_adjusted_density(model, density):
    non_shared_params = 0
    count_layer = 0
    for name, param in model.named_parameters():
        if ('weight' in name and
                ('conv' in name.lower() or 'linear' in name.lower() or 'classifier' in name.lower() or name.endswith(
                    '.shortcut.0.weight')) and
                'bn' not in name.lower() and
                'relu' not in name.lower()):
            non_shared_params += param.numel()
            count_layer += 1
            print(f"Non-shared layer: {name}, params: {param.numel()}")
        # else:
        #     print(f"Skipped layer: {name}, params: {param.numel()}")

    shared_params = model.get_shared_para()

    total_params = non_shared_params + shared_params

    print(f"Total params: {total_params}")
    print(f"Non-shared params: {non_shared_params}")
    print(f"Shared params: {shared_params}")

    adjusted_density = total_params * density / non_shared_params

    return adjusted_density


def track_gradient_flow(model):
    grad_norms = {
        'Conv1': 0.0,
        'Layer1': 0.0,
        'Layer2': 0.0,
        'Layer3': 0.0,
        'Layer4': 0.0,
        'classifier': 0.0,
        'Total': 0.0
    }
    count = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if ('weight' in name and
                    ('conv' in name.lower() or 'linear' in name.lower() or 'classifier' in name.lower() or name.endswith('.shortcut.0.weight')) and
                    'bn' not in name.lower() and
                    'relu' not in name.lower()):
                count += 1
                grad_norm = param.grad.norm(2).item()
                grad_norms['Total'] += grad_norm

                if name == 'conv1.weight':
                    grad_norms['Conv1'] += grad_norm
                elif 'layer1' in name:
                    grad_norms['Layer1'] += grad_norm
                elif 'layer2' in name:
                    grad_norms['Layer2'] += grad_norm
                elif 'layer3' in name:
                    grad_norms['Layer3'] += grad_norm
                elif 'layer4' in name:
                    grad_norms['Layer4'] += grad_norm
                elif 'classifier' in name:
                    grad_norms['classifier'] += grad_norm
                # print('debug: ',name)

    print(f'counted {count} meaningful layers')
    print("Gradient norms:", grad_norms)
    return grad_norms


def inference_speed(model, device, test_loader, num_runs=100, batch_size=None):
    model.eval()
    model.to(device)

    if batch_size is None:
        batch_size = test_loader.batch_size

    # Latency measurement
    latency_times = []
    with torch.no_grad():
        for i in range(num_runs):
            data, _ = next(iter(test_loader))
            data = data[:batch_size].to(device)

            start_time = time.time()
            _ = model(data)
            end_time = time.time()

            latency_times.append(end_time - start_time)

    avg_latency = sum(latency_times) / len(latency_times)
    print(f"Average inference latency (batch size {batch_size}): {avg_latency:.6f} seconds")

    # Throughput measurement
    start_time = time.time()
    total_samples = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _ = model(data)
            total_samples += data.size(0)
    end_time = time.time()

    throughput = total_samples / (end_time - start_time)
    print(f"Inference throughput: {throughput:.2f} samples/second")

    return avg_latency, throughput


def train(args, model, device, train_loader, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                       100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))

    # training summary
    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary',
        train_loss / batch_idx, correct, n, 100. * correct / float(n)))

    return train_loss / batch_idx, correct / float(n)


def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return test_loss, correct / float(n)


def eval_imagenet(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct5 = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
            correct5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            n += target.size(0)

    test_loss /= n
    top1_acc = 100. * correct1 / n
    top5_acc = 100. * correct5 / n

    print_and_log(
        '\n{}: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.2f}%), Top-5 Accuracy: {}/{} ({:.2f}%)\n'.format(
            'Test evaluation' if is_test_set else 'Evaluation',
            test_loss, correct1, n, top1_acc, correct5, n, top5_acc))

    return test_loss, top1_acc, top5_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1,
                        help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true',
                        help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true',
                        help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--inference-speed', action='store_true')
    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    # drelu settings
    parser.add_argument('--disable_drelu_grad', action='store_true', help='disable_drelu_grad')
    parser.add_argument('--start_ghost_epoch', type=int, default=None, help='Start epoch for gradual phasing')
    parser.add_argument('--end_ghost_epoch', type=int, default=None, help='End epoch for gradual phasing')
    # weight sharing
    parser.add_argument('--ratio', type=int, default=2)
    # cyclic sparsity
    parser.add_argument('--cyclic', action='store_true')
    parser.add_argument('--num_cycles', type=int, default=2)
    parser.add_argument('--cyclic-length', type=int, default=50)
    parser.add_argument('--cyclic-length-multiplier', type=float, default=1.0)
    parser.add_argument('--density_max_multiplier', type=int, default=3)
    parser.add_argument('--cyclic-pattern', type=str, default='cosine', choices=['cosine', 'triangular'])
    # imagenet
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--wandb-mode', type=str, choices=("dryrun", "online"), default="dryrun")
    parser.add_argument('--wandb-project', type=str, default='ES2')


    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.wandb_mode == "dryrun":
        wandb.init(mode="dryrun")
    elif args.wandb_mode == "online":
        wandb.init(project="ES2", entity="tensorstrike", config=vars(args))

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('=' * 80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))
        c = 10
        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                              max_threads=args.max_threads)
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split,
                                                                               max_threads=args.max_threads)
            c = 100
        elif args.data == 'imagenet':
            train_loader, valid_loader = get_imagenet_dataloaders(args)
            test_loader = valid_loader
            c = 1000

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        elif args.model == 'ResNet18':
            model = ResNet18(c=c, ratio=args.ratio).to(device)
        elif args.model == 'ResNet34':
            model = ResNet34(c=c, ratio=args.ratio).to(device)
            # model = ResNet34_ReLU(c=c, ratio=args.ratio).to(device)
        elif args.model == 'ResNet50':
            if args.data == 'imagenet':
                model = ResNet50_ImageNet(c=1000, ratio=args.ratio).to(device)
            else:       # cifar
                model = ResNet50(c=c, ratio=args.ratio).to(device)

        else:
            cls, cls_args = models[args.model]
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)

        print_and_log(model)
        print_and_log('=' * 60)
        print_and_log(args.model)
        print_and_log('=' * 60)

        print_and_log('=' * 60)
        print_and_log('Prune mode: {0}'.format(args.death))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('=' * 60)

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2,
                                  nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        milestones = [int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, last_epoch=-1)
        if args.start_ghost_epoch is None:
            args.start_ghost_epoch = milestones[0] / 2
        if args.end_ghost_epoch is None:
            args.end_ghost_epoch = milestones[0] - 1

        start_ghost_epoch = args.start_ghost_epoch
        end_ghost_epoch = args.end_ghost_epoch

        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_and_log("=> loaded checkpoint '{}' (epoch {})"
                              .format(args.resume, checkpoint['epoch']))
                print_and_log('Testing...')
                evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.resume))

        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=None,
                                       dynamic_loss_scale=True,
                                       dynamic_loss_args={'init_scale': 2 ** 16})
            model = model.half()

        mask = None
        if args.sparse:

            decay = CosineDecay(args.death_rate, len(train_loader) * (args.epochs * args.multiplier))
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay,
                           growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args)
            density = calculate_adjusted_density(model, args.density)  # we adjust density to account for weight sharing
            if args.cyclic:
                steps_1st_cycle = args.cyclic_length * len(train_loader)
                steps_per_cycle=[]
                for i in range(args.num_cycles):
                    steps_per_cycle.append(steps_1st_cycle * args.cyclic_length_multiplier ** i)

                mask.steps_per_cycle = steps_per_cycle
                mask.cyclic_end_step = sum(steps_per_cycle)


            mask.add_module(model, sparse_init=args.sparse_init, density=density)

        if args.cyclic:
            record_epoch = max(mask.cyclic_end_step/len(train_loader), end_ghost_epoch)  # record best model after this epoch
        else:
            record_epoch = end_ghost_epoch
        best_acc = 0.0

        # for epoch in range(1, args.epochs*args.multiplier + 1):
        #     t0 = time.time()
        #     train(args, model, device, train_loader, optimizer, epoch, mask)
        #     lr_scheduler.step()
        #     if args.valid_split > 0.0:
        #         val_acc = evaluate(args, model, device, valid_loader)

        #     if val_acc > best_acc:
        #         print('Saving model')
        #         best_acc = val_acc
        #         torch.save(model.state_dict(), args.save)

        #     print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))

        for epoch in range(1, args.epochs * args.multiplier + 1):
            t0 = time.time()

            if args.disable_drelu_grad:
                if epoch == start_ghost_epoch:  # when we freeze gradients of drelu
                    print("Disabling grad for DyReLU")
                    for name, module in model.named_modules():
                        if isinstance(module, DyReLUB):
                            for param in module.parameters():
                                param.requires_grad = False
                                param.grad = None
                                if param in optimizer.state:
                                    if 'momentum_buffer' in optimizer.state[param]:
                                        optimizer.state[param]['momentum_buffer'] = torch.zeros_like(param)

                if start_ghost_epoch <= epoch <= end_ghost_epoch:  # decay factor (beta) for phasing out drelu
                    decay_factor = 1 - (epoch - start_ghost_epoch) / (end_ghost_epoch - start_ghost_epoch)
                    print(f"Decay factor for epoch {epoch}: {decay_factor}")
                    for name, module in model.named_modules():
                        if isinstance(module, DyReLUB):
                            module.beta = decay_factor

            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, mask)
            gradient_norms = track_gradient_flow(model)
            # grad_norms_flat = {f"Gradient Norm/{layer}": norm for layer, norm in gradient_norms.items()}
            grad_norms_flat = {f"Gradient Norm/{layer_group}": norm for layer_group, norm in gradient_norms.items()}


            lr_scheduler.step()

            if args.data == 'imagenet':
                val_loss, val_top1_acc, val_top5_acc = eval_imagenet(args, model, device, valid_loader)
            else:
                val_loss, val_acc = evaluate(args, model, device, valid_loader)

            if epoch >= record_epoch:
                if args.data == 'imagenet':
                    if val_top1_acc > best_acc:
                        print('Saving model', args.save)
                        best_acc = val_top1_acc
                        torch.save(model.state_dict(), args.save)
                else:
                    if val_acc > best_acc:
                        print('Saving model', args.save)
                        best_acc = val_acc
                        torch.save(model.state_dict(), args.save)


            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(
                optimizer.param_groups[0]['lr'], time.time() - t0))

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                # "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                # "gradient_norm": grad_norms_flat,
            }
            metrics.update(grad_norms_flat)

            if args.data == 'imagenet':
                metrics.update({
                    "val_top1_accuracy": val_top1_acc,
                    "val_top5_accuracy": val_top5_acc,
                })
            else:
                metrics.update({
                    "val_accuracy": val_acc,
                })

            if args.sparse:
                sparse_metrics = mask.get_metrics()
                metrics.update(sparse_metrics)

            wandb.log(metrics)

        print('Testing')

        if args.data == 'imagenet':
            eval_imagenet(args, model, device, test_loader, is_test_set=True)
            if args.inference_speed == True:
                inference_speed(model, device, test_loader, num_runs=100, batch_size=2)

        else:       # cifar10/100
            evaluate(args, model, device, test_loader, is_test_set=True)
            if args.inference_speed == True:
                inference_speed(model, device, test_loader, num_runs=100, batch_size=128)


        print_and_log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))

        # layer_fired_weights, total_fired_weights = mask.fired_masks_update()
        # for name in layer_fired_weights:
        #     print('The final percentage of fired weights in the layer', name, 'is:', layer_fired_weights[name])
        # print('The final percentage of the total fired weights is:', total_fired_weights)


if __name__ == '__main__':
    main()