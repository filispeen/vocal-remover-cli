import click
from logging import getLogger
#from f_voice-remover import __version__
from subprocess import run

LOG = getLogger(__name__)

class RichHelpFormatter(click.HelpFormatter):
    def __init__(
        self,
        indent_increment: int = 2,
        width: int | None = None,
        max_width: int | None = None,
    ) -> None:
        width = 100
        super().__init__(indent_increment, width, max_width)

def patch_wrap_text():
    orig_wrap_text = click.formatting.wrap_text
    def wrap_text(
        text,
        width=78,
        initial_indent="",
        subsequent_indent="",
        preserve_paragraphs=False,
    ):
        return orig_wrap_text(
            text.replace("\n", "\n\n"),
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            preserve_paragraphs=True,
        ).replace("\n\n", "\n")

    click.formatting.wrap_text = wrap_text

patch_wrap_text()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
click.Context.formatter_class = RichHelpFormatter

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """
    To train a model, run train.\n
    To infer a model, run infer.
    """

@cli.command()
@click.option(
    "--gpu",
    "-g",
    type=int,
    default=-1,
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=2019
)
@click.option(
    "-sr",
    "-r",
    type=int,
    default=44100
)
@click.option(
    "--hop_length",
    "-h",
    type=int,
    default=1024
)
@click.option(
    "--n_fft",
    "-f",
    type=int,
    default=2048
)
@click.option(
    "--dataset",
    "-d",
    required=True
)
@click.option(
    "--split_mode",
    "-S",
    type=click.Choice(["random", "subdirs"]),
    default="random"
)
@click.option(
    "--learning_rate",
    "-l",
    type=float,
    default=0.001
)
@click.option(
    "--lr_min",
    type=float,
    default=0.0001
)
@click.option(
    "--lr_decay_factor",
    type=float,
    default=0.9
)
@click.option(
    "--lr_decay_patience",
    type=int,
    default=6
)
@click.option(
    "--batchsize",
    "-B",
    type=int,
    default=4
)
@click.option(
    "--accumulation_steps",
    "-A",
    type=int,
    default=1
)
@click.option(
    "--cropsize",
    "-C",
    type=int,
    default=256
)
@click.option(
    "--patches",
    "-p",
    type=int,
    default=16
)
@click.option(
    "--val_rate",
    "-v",
    type=float,
    default=0.2
)
@click.option(
    "--val_filelist",
    "-V",
    type=str,
    default=None
)
@click.option(
    "--val_batchsize",
    "-b",
    type=int,
    default=6
)
@click.option(
    "--val_cropsize",
    "-c",
    type=int,
    default=256
)
@click.option(
    "--num_workers",
    "-w",
    type=int,
    default=6
)
@click.option(
    "--epoch",
    "-E",
    type=int,
    default=200
)
@click.option(
    "--reduction_rate",
    "-R",
    type=float,
    default=0.0
)
@click.option(
    "--reduction_level",
    "-L",
    type=float,
    default=0.2
)
@click.option(
    "--mixup_rate",
    "-M",
    type=float,
    default=0.0
)
@click.option(
    "--mixup_alpha",
    "-a",
    type=float,
    default=1.0
)
@click.option(
    "--pretrained_model",
    "-P",
    type=str,
    default=None
)
def train(**kwargs):
    print(kwargs)
    run(["python", "train.py", "--gpu", str(kwargs['gpu']), "--dataset", str(kwargs['dataset']), '--seed', str(kwargs['seed']), '-r', str(kwargs['r']), '--hop_length', str(kwargs['hop_length']), '--n_fft', str(kwargs['n_fft']), '--split_mode', str(kwargs['split_mode']), '--learning_rate', str(kwargs['learning_rate']), '--lr_min', str(kwargs['lr_min']), '--lr_decay_factor', str(kwargs['lr_decay_factor']), '--lr_decay_patience', str(kwargs['lr_decay_patience']), '--batchsize', str(kwargs['batchsize']), '--accumulation_steps', str(kwargs['accumulation_steps']), '--cropsize', str(kwargs['cropsize']), '--patches', str(kwargs['patches']), '--val_rate', str(kwargs['val_rate']), '--val_filelist', str(kwargs['val_filelist']), '--val_batchsize', str(kwargs['val_batchsize']), '--val_cropsize', str(kwargs['val_cropsize']), '--num_workers', str(kwargs['num_workers']), '--epoch', str(kwargs['epoch']), '--reduction_rate', str(kwargs['reduction_rate']), '--reduction_level', str(kwargs['reduction_level']), '--mixup_rate', str(kwargs['mixup_rate']), '--mixup_alpha', str(kwargs['mixup_alpha']), '--pretrained_model', str(kwargs['pretrained_model'])])

@cli.command()
@click.option(
    "--gpu",
    "-g",
    type=int,
    default=-1
)
@click.option(
    "--pretrained_model",
    "-P",
    type=str,
    required=True
)
@click.option(
    "--input",
    "-i",
    type=str,
    required=True
)
@click.option(
    "-sr",
    "-r",
    type=int,
    default=44100
)
@click.option(
    "--hop_length",
    "-h",
    type=int,
    default=1024
)
@click.option(
    "--n_fft",
    "-f",
    type=int,
    default=2048
)
@click.option(
    "--batchsize",
    "-B",
    type=int,
    default=4
)
@click.option(
    "--cropsize",
    "-C",
    type=int,
    default=256
)
@click.option(
    "--output_image",
    "-I",
    is_flag=True,
    help="store_true"
)
@click.option(
    "--postprocess",
    "-p",
    is_flag=True,
    help="store_true"
)
@click.option(
    "--tta",
    "-t",
    is_flag=True,
    help="store_true"
)
@click.option(
    "--output_dir",
    "-o",
    type=str,
    default=""
)
def infer(**kwargs):
    print(kwargs)
    run(["python", "inference.py", "--gpu", str(kwargs['gpu']), '-r', str(kwargs['r']), '--hop_length', str(kwargs['hop_length']), '--n_fft', str(kwargs['n_fft']), '--batchsize', str(kwargs['batchsize']), '--cropsize', str(kwargs['cropsize']), "--output_image", str(kwargs["output_image"]), "--postprocess", str(kwargs["postprocess"]), "--tta", str(kwargs["tta"]), "--output_dir", str(kwargs["output_dir"])])

if __name__ == '__main__':
    cli()
