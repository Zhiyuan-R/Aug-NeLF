import torch
import configargparse
import shutil
import os

from torch.utils.tensorboard import SummaryWriter

from models import NeLFNet, Adv_NeLFNet
from data import RayNeLFDataset, RayBatchCollater, ExhibitNeRFDataset

def create_arg_parser():
    parser = configargparse.ArgumentParser()
    
    # needed args
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--expdir', type=str,
                        help='experiment name')
    
    # basic args
    parser.add_argument('--gpuid', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--logdir', type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluate without training')
    parser.add_argument("--num_workers", type=int, default=8,
                help='number of workers used for data loading')
    
    ## data args
    parser.add_argument("--data_path", "--datadir", type=str,
                        help='input data directory')

    # training option
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='exponential learning rate decay scale')
    parser.add_argument('--decay_step', type=float, default=250,
                        help='exponential learning rate decay iteration (in 1000 steps)')
    parser.add_argument("--batch_size", "--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--pin_mem", action='store_true', default=True,
                    help='turn on pin memory for data loading')
    parser.add_argument("--max_steps", "--N_iters", type=int, default=50000, 
                        help='max iteration number (number of iteration to finish training)')
    
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
    
    # create testset and exhibitset
    print("Loading nerf data:", args.data_path)
    test_set = RayNeRFDataset(args.data_path, split='test')
    try:
        exhibit_set = ExhibitNeRFDataset(args.data_path)
    except FileNotFoundError:
        exhibit_set = None
        print("Warning: No exhibit set!")
    
    ##### Training Stage #####
    if not args.eval:
        train_set = RayNeLFDataset(root_dir = arg.data_path,
                                   split = 'train')
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   collate_fn=RayBatchCollater(),
                                                   num_workers=args.num_workers,
                                                   pin_memory=args.pin_mem)

        # Summary writers
        summary_writer = SummaryWriter(log_dir=log_dir)
        print("Starting training ...")
        while global_step < args.max_steps:
            
            if args.adv:
            
                pass
            
            else:
                global_step = train_one_epoch(model, optimizer, scheduler,
                                              train_loader, test_set, exhibit_set,
                                              summary_writer,
                                              global_step,
                                              args.max_steps,
                                              run_dir,
                                              device)
                
    ##### Testing Stage #####
    save_dir = os.path.join(run_dir, 'test')
    os.makedirs(save_dir, exist_ok=True)
                
if __name__=='__main__':

    # read arguments and configs
    parser = create_arg_parser()
    args, _ =  parser.parse_known_args()
    
    main(args)
