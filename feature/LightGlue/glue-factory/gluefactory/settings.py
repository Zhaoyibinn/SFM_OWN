from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
DATA_PATH = Path("/media/zhaoyibin/3DRE/LGDATA/")  # datasets and pretrained weights,存放数据集位置
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = root / "outputs/results/"  # evaluation results
