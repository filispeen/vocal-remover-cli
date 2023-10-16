import argparse
from subprocess import run

def add_common_arguments(parser):
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--sr', '-r', type=int, default=44100)
    parser.add_argument('--hop_length', '-H', type=int, default=1024)
    parser.add_argument('--n_fft', '-f', type=int, default=2048)
    parser.add_argument('--batchsize', '-B', type=int, default=4)
    parser.add_argument('--cropsize', '-c', type=int, default=256)
    parser.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')

def main():
    parser = argparse.ArgumentParser(description='CLI для вашего vc-remover пакета')

    parser.add_argument('command', choices=['train', 'infer'], help='Команда (train или infer)')

    args = parser.parse_args()

    # Общие аргументы для обоих команд
    add_common_arguments(parser)

    if args.command == 'train':
        # Добавьте аргументы для train.py
        parser.add_argument('--seed', '-s', type=int, default=2019)
        parser.add_argument('--dataset', '-d', required=True)
        parser.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
        parser.add_argument('--learning_rate', '-l', type=float, default=0.001)
        parser.add_argument('--lr_min', type=float, default=0.0001)
        parser.add_argument('--lr_decay_factor', type=float, default=0.9)
        parser.add_argument('--lr_decay_patience', type=int, default=6)
        parser.add_argument('--patches', '-p', type=int, default=16)
        parser.add_argument('--val_rate', '-v', type=float, default=0.2)
        parser.add_argument('--val_filelist', '-V', type=str, default=None)
        parser.add_argument('--val_batchsize', '-b', type=int, default=6)
        parser.add_argument('--num_workers', '-w', type=int, default=6)
        parser.add_argument('--epoch', '-E', type=int, default=200)
        parser.add_argument('--reduction_rate', '-R', type=float, default=0.0)
        parser.add_argument('--reduction_level', '-L', type=float, default=0.2)
        parser.add_argument('--mixup_rate', '-M', type=float, default=0.0)
        parser.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
        parser.add_argument('--debug', action='store_true')
    elif args.command == 'infer':
        # Добавьте аргументы для infer.py
        parser.add_argument('--input', '-i', required=True)
        parser.add_argument('--output_image', '-I', action='store_true')
        parser.add_argument('--postprocess', '-p', action='store_true')
        parser.add_argument('--tta', '-t', action='store_true')
        parser.add_argument('--output_dir', '-o', type=str, default="")

    args = parser.parse_args()

    # Обработка аргументов для выбранной команды
    if args.command == 'train':
        # Вызов train.py с аргументами
        run(['python', 'train.py', '--gpu', str(args.gpu), '--seed', str(args.seed), ...])
    elif args.command == 'infer':
        # Вызов infer.py с аргументами
        run(['python', 'infer.py', '--gpu', str(args.gpu), '--pretrained_model', args.pretrained_model, ...])

if __name__ == '__main__':
    main()
