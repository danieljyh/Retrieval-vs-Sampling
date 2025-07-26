from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class Args():
    model: str = field(
        default=None,
        metadata={}
    )
    method: str = field(
        default=None,
        metadata={}
    )
    tasks: List[str] = field(
        default_factory=lambda: ["qaego4d", "egoschema", "cgbench", "mlvu", "activitynet_qa", "rvs_ego", "rvs_movie"],
        metadata={'help': 'Which dataset o evaluate?'}
    )
    load_results: bool = field(
        default=False,
    )


    sample_fps: float = field(
        default=0.5
    )
    retrieve_size: int = field(
        default=64
    )
    n_local: int = field(
        default=15000
    )


    dataset_dir: str = field(
        default="data/",
        metadata={'help': 'The evaluation json data path.'}
    )
    result_dir: Optional[str] = field(
        default="results/",
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )
    prefix_dir: str = field(
        default="",
        metadata={'help': "Directory in which the result json file get stored"}
    )
    postfix: str = field(
        default="",
        metadata={'help': "Append given postfix to the result json file"}
    )


    newline_as_eos: bool = field(
        default=True,
        metadata={'help': 'Whether to use new line as eos (for QA tasks only) or not.'}
    )
    # max_length: int = field(
    #     default=31500,
    #     metadata={'help': 'Max input length.'}
    # )
    # truncate_from_middle: bool = field(
    #     default=True,
    #     metadata={'help': 'Truncate inputs from the middle.'}
    # )

