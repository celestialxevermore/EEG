from config import arg
from data_loader import target_data_loader, source_data_loader
from build_net import build_net
from make_solver import make_solver
from utils import control_random, timeit

@timeit
def main():
    args = arg()

    # seed control
    if args.seed:
        control_random(args)

    # load datasets
    if args.mode == "pretrain":
        train_loader, val_loader, test_loader = source_data_loader(args)
        test_loader = None
    elif args.mode == "finetune":
        train_loader, val_loader, test_loader = target_data_loader(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # import backbone model
    net = build_net(args, train_loader.dataset.X.shape)

    # make solver (runner)
    solver = make_solver(args, net, train_loader, val_loader, test_loader)

    # train or finetune
    solver.experiment()

if __name__ == '__main__':
    main()
