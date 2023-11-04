import click
from logging import getLogger
from pathlib import Path
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
    Training own mdx-net model not availbe by originial developers of UVR. Maybe soon....
    To infer a model, run infer.\n
    To enseble infer a model, run ensemble_infer.
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
    "--mixture_dataset",
    "-m",
    type=click.Path(),
    required=True
)
@click.option(
    "--instrumental_dataset",
    "-i",
    type=click.Path(),
    required=True
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
    "--cropsize",
    "-c",
    type=int,
    default=256
)
@click.option(
    "--val_rate",
    "-v",
    type=float,
    default=0.1
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
    default=4
)
@click.option(
    "--val_cropsize",
    "-c",
    type=int,
    default=512
)
@click.option(
    "--patches",
    "-p",
    type=int,
    default=16
)
@click.option(
    "--epoch",
    "-E",
    type=int,
    default=100
)
@click.option(
    "--inner_epoch",
    "-e",
    type=int,
    default=4
)
@click.option(
    "--oracle_rate",
    "-O",
    type=float,
    default=0
)
@click.option(
    "--oracle_drop_rate",
    "-o",
    type=float,
    default=0.5
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
    type=click.Path(),
    default=None
)
@click.option(
    "--debug",
    is_flag=True,
    default=False
)
def train(**kwargs):
    print(kwargs)
    raise RuntimeWarning("Training unavailable. But you can inference your song with already available models.")
    #from train import train_start
    #train_start(a_seed=kwargs['seed'], a_pretrained_model=kwargs['pretrained_model'], a_gpu=kwargs['gpu'], a_learning_rate=kwargs['learning_rate'], a_lr_decay_factor=kwargs['lr_decay_factor'], a_lr_decay_patience=kwargs['lr_decay_patience'], a_lr_min=kwargs['lr_min'], a_mixture_dataset=kwargs['mixture_dataset'], a_instrumental_dataset=kwargs['instrumental_dataset'], a_val_rate=kwargs['val_rate'], a_val_filelist=kwargs['val_filelist'], a_debug=kwargs['debug'], a_val_cropsize=kwargs['val_cropsize'], a_sr=kwargs['r'], a_hop_length=kwargs['hop_length'], a_val_batchsize=kwargs['val_batchsize'], a_epoch=kwargs['epoch'], a_cropsize=kwargs['cropsize'], a_patches=kwargs['patches'], a_mixup_rate=kwargs['mixup_rate'], a_mixup_alpha=kwargs['mixup_alpha'], a_inner_epoch=kwargs['inner_epoch'], a_batchsize=kwargs['batchsize'], a_oracle_rate=kwargs['oracle_rate'], a_oracle_drop_rate=kwargs['oracle_drop_rate'])
    #run(["python", "train.py", "--gpu", str(kwargs['gpu']), "--dataset", str(kwargs['dataset']), '--seed', str(kwargs['seed']), '-r', str(kwargs['r']), '--hop_length', str(kwargs['hop_length']), '--n_fft', str(kwargs['n_fft']), '--split_mode', str(kwargs['split_mode']), '--learning_rate', str(kwargs['learning_rate']), '--lr_min', str(kwargs['lr_min']), '--lr_decay_factor', str(kwargs['lr_decay_factor']), '--lr_decay_patience', str(kwargs['lr_decay_patience']), '--batchsize', str(kwargs['batchsize']), '--accumulation_steps', str(kwargs['accumulation_steps']), '--cropsize', str(kwargs['cropsize']), '--patches', str(kwargs['patches']), '--val_rate', str(kwargs['val_rate']), '--val_filelist', str(kwargs['val_filelist']), '--val_batchsize', str(kwargs['val_batchsize']), '--val_cropsize', str(kwargs['val_cropsize']), '--num_workers', str(kwargs['num_workers']), '--epoch', str(kwargs['epoch']), '--reduction_rate', str(kwargs['reduction_rate']), '--reduction_level', str(kwargs['reduction_level']), '--mixup_rate', str(kwargs['mixup_rate']), '--mixup_alpha', str(kwargs['mixup_alpha']), '--pretrained_model', str(kwargs['pretrained_model'])])

@cli.command()
@click.option(
    "--gpu",
    "-g",
    type=int,
    default=-1,
)
@click.option(
    "--pretrained_model",
    "-P",
    type=click.Path(),
    default=None
)
@click.option(
    "--input",
    "-i",
    type=click.Path(),
    required=True
)
@click.option(
    "--nn_architecture",
    "-n",
    type=click.Choice(['default', '33966KB', '123821KB', '129605KB', '537238KB']),
    default='default'
)
@click.option(
    "--model_params",
    "-m",
    type=str,
    default=''
)
@click.option(
    "--window_size",
    "-w",
    type=int,
    default=512
)
@click.option(
    "--output_image",
    "-I",
    is_flag=True,
    default=False
)
@click.option(
    "--deepextraction",
    "-D",
    is_flag=True,
    default=False
)
@click.option(
    "--postprocess",
    "-p",
    is_flag=True,
    default=False
)
@click.option(
    "--is_vocal_model",
    "-vm",
    is_flag=True,
    default=False
)
@click.option(
    "--tta",
    "-t",
    is_flag=True,
    default=False
)
@click.option(
    "--high_end_process",
    "-H",
    type=click.Choice(['none', 'bypass', 'correlation', 'mirroring', 'mirroring2']),
    default='none'
)
@click.option(
    "--aggressiveness",
    "-A",
    type=float,
    default=0.07,
)
def infer(**kwargs):
    print(kwargs)
    from inference import start_infer
    start_infer(a_nn_architecture=kwargs['nn_architecture'], a_model_params=kwargs['model_params'], a_pretrained_model=kwargs['pretrained_model'], a_gpu=kwargs['gpu'], a_input=kwargs['input'], a_high_end_process=kwargs['high_end_process'], a_window_size=kwargs['window_size'], a_tta=kwargs['tta'], a_aggressiveness=kwargs['aggressiveness'], a_postprocess=kwargs['postprocess'], a_is_vocal_model=kwargs['is_vocal_model'], a_output_image=kwargs['output_image'], a_deepextraction=kwargs['deepextraction'])
    #run(["python", "inference.py", "--gpu", str(kwargs['gpu']), '-r', str(kwargs['r']), '--hop_length', str(kwargs['hop_length']), '--n_fft', str(kwargs['n_fft']), '--batchsize', str(kwargs['batchsize']), '--cropsize', str(kwargs['cropsize']), "--output_image", str(kwargs["output_image"]), "--postprocess", str(kwargs["postprocess"]), "--tta", str(kwargs["tta"]), "--output_dir", str(kwargs["output_dir"])])

@cli.command()
def ensemble_infer(**kwargs):
    print(kwargs)

if __name__ == '__main__':
    cli()
