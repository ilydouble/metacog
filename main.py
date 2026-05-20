import os
import argparse
from ds1000_reflexion import run_ds1000_reflexion, run_ds1000_simple
from immediate_refinement import run_immediate_refinement
from immediate_reflexion import run_immediate_reflexion

from simple import run_simple
from reflexion import run_reflexion
from reflexion_ucs import run_reflexion_ucs
from test_acc import run_test_acc
from utils import read_jsonl, read_jsonl_gz, make_printv
from rich.panel import Panel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_name", type=str, help="日志文件名", default=None)
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `reflexion`")
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`")
    parser.add_argument("--dataset_type", type=str,
                        help="Dataset adapter: `humaneval` or `ds1000`", default="humaneval")
    parser.add_argument(
        "--model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--expansion_factor", type=int,
                        help="The expansion factor for the reflexion UCS and A* strategy", default=3)

    parser.add_argument("--is_leetcode", action='store_true',
                        help="To run the leetcode benchmark")  # Temporary

    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    # TODO: implement this
    # parser.add_argument("--is_resume", action='store_true', help="To resume run")
    # parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple, delete_keys=["expansion_factor", "max_iters"])
    elif strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "immediate-reflexion":
        return kwargs_wrapper_gen(run_immediate_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "immediate-refinement":
        return kwargs_wrapper_gen(run_immediate_refinement, delete_keys=["expansion_factor"])
    elif strategy == "reflexion-ucs":
        return kwargs_wrapper_gen(run_reflexion_ucs)
    elif strategy == "test-acc":
        return kwargs_wrapper_gen(run_test_acc, delete_keys=["expansion_factor", "max_iters"])
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


def run_ds1000_strategy(args, dataset, log_path):
    if args.strategy == "simple":
        return run_ds1000_simple(
            dataset=dataset,
            model_name=args.model,
            pass_at_k=args.pass_at_k,
            log_path=log_path,
            verbose=args.verbose,
        )
    elif args.strategy == "reflexion":
        return run_ds1000_reflexion(
            dataset=dataset,
            model_name=args.model,
            max_iters=args.max_iters,
            pass_at_k=args.pass_at_k,
            log_path=log_path,
            verbose=args.verbose,
        )
    else:
        raise ValueError("DS1000 adapter currently supports `simple` and `reflexion` only")


def main(args):
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # get the dataset name
    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    # check if log path already exists
    log_dir = os.path.join(args.root_dir, args.run_name)
    # log_path = os.path.join(
    #     log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")
    log_path = os.path.join(log_dir, f"{args.log_name}.jsonl")
    stdout_log_path = os.path.join(log_dir, f"{args.log_name}.log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # check if the strategy is valid
    run_strategy = strategy_factory(args.strategy)

    # print starting message
    print_v = make_printv(args.verbose, stdout_log_path)
    if args.verbose:
        print_v(Panel(f"Strategy: [bold]{args.strategy}[/bold]\nPass@k: [bold]{args.pass_at_k}[/bold]", title="Configuration", border_style="green"))
    else:
        print(f"Logs will be saved in `{log_dir}`")

    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")
    if args.dataset_type == "ds1000":
        run_ds1000_strategy(args, dataset, log_path)
        # print(f"Done! Check out the logs in `{log_path}`")
        print_v(f"Done! Check out the logs in `{log_path}`")
        return
    elif args.dataset_type != "humaneval":
        raise ValueError(f"Dataset adapter `{args.dataset_type}` is not supported")

    # start the run
    # evaluate with pass@k
    run_strategy(
        dataset=dataset,
        model_name=args.model,
        language=args.language,
        max_iters=args.max_iters,
        pass_at_k=args.pass_at_k,
        log_path=log_path,
        verbose=args.verbose,
        expansion_factor=args.expansion_factor,
        is_leetcode=args.is_leetcode
    )

    # print(f"Done! Check out the logs in `{log_path}`")
    print_v(f"Done! Check out the logs in `{log_path}`")


if __name__ == "__main__":
    args = get_args()
    main(args)
