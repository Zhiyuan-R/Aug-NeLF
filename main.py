import torch
import configargparse
import shutil
import os

from models.nelf import NeLFNet
from models.adv_nelf import Adv_NeLFNet

def create_arg_parser():
    parser = configargparse.ArgumentParser()
    
    # needed args
    parser.add_argument('--expdir', type=str,
                        help='experiment name')
    
    # basic args
    parser.add_argument('--gpuid', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--logdir', type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluate without training')
    
    # training option
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='exponential learning rate decay scale')
    parser.add_argument('--decay_step', type=float, default=250,
                        help='exponential learning rate decay iteration (in 1000 steps)')
    
    # hyper-parameter for adversarial training
    parser.add_argument('--adv', action='store_true',
                        help='turn on adv training!')

    return parser

def main(args):
    
    # specify the device
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() \
                                                else 'cpu')
    
    # create log and specify the path to save the log
    run_dir  = os.path.join(args.logdir, args.expdir)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'tensorboard')
    
    # make the path for run 
    if not os.path.exists(run_dir):
        if not args.eval:
            os.makedirs(run_dir)
            os.makedirs(ckpt_dir)
            os.makedirs(log_dir)
            
            # dump training configuration
            config_path = os.path.join(run_dir, 'args_log.txt')
            parser.write_config_file(args, [config_path])
        else:
            print('You want to evaluate but you have not even train it!!!')
            return
        
    # create the model and optimizer
    if args.adv:
        model = Adv_NeLFNet().to(device)
    else:
        model = NeLFNet().to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), \
                                    lr=args.lr, \
                                    betas=(0.9, 0.999))
    scheduler = LRScheduler(optimizer=optimizer, \
                            init_lr=args.lr, \
                            decay_rate=args.decay_rate, \
                            decay_steps=args.decay_step*1000)
    global_step = 0
    
    # create test and exhibit set dataset
    
    import pdb
    pdb.set_trace()
    
    pass

if __name__=='__main__':

    # read arguments and configs
    parser = create_arg_parser()
    args =  parser.parse_args()
    
    main(args)
