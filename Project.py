from dataclasses import dataclass
from pathlib import Path
import typing


@dataclass
class Project:
    """
    This class represents our project. It stores useful information about the structure, e.g. paths.
    """

    base_dir: Path = Path(__file__).parents[0]
    data_dir = base_dir / 'dataset'
    checkpoint_dir = base_dir / 'checkpoint'

    cli_commands: typing.Tuple[str] = ('data-download', 'data-preprocess', 'data-inspect-train', 'data-inspect-val', 'data-inspect-test', 'train', 'test', 'infer', 'scrap', )
    labels: typing.Tuple[int] = (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,)

    def __post_init__(self):
        # create the directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
