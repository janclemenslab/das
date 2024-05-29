"""
A bit on semantics:
- blocks are parts of the data (individual files in a file list, parts of a data array)
- individual blocks are assigned specific groups (train/test/val)
"""

from typing import Optional, Dict, List, Any, Tuple, TypeVar
import numpy as np
import pandas as pd

Block = TypeVar("Block")
Group = TypeVar("Group")


def groupstats(stats: List, groups: List[Group]):
    """_summary_

    Args:
        stats (_type_): stats [x] for each group [n, x]
        groups (_type_): ids for each group [n,]

    Returns:
        Dict[group_id, x]: stats average over all members of groups
    """
    unique_groups, group_index = np.unique(groups, return_inverse=True)

    group_stats = {}
    for index, group in enumerate(unique_groups):
        group_stats[group] = np.mean(stats[group_index == index, :], axis=0)
    return group_stats


def group_splits(data: np.ndarray, group_sizes: List[str]) -> List[np.ndarray]:
    group_cdf = np.insert(np.cumsum(group_sizes), 0, 0)[:-1]
    split_points = (group_cdf * data.shape[0]).astype(int)
    blocks = np.array_split(data, split_points[1:])

    split_points = list(split_points)
    split_points.append(data.shape[0])
    split_points = [(start, end) for start, end in zip(split_points[:-1], split_points[1:])]

    return blocks, split_points


def group_blocks(nb_blocks: int, group_sizes: List[str], group_names: List[Group]):
    group_cdf = np.insert(np.cumsum(group_sizes), 0, 0)[:-1]
    group_probs = np.linspace(0, 1, nb_blocks)
    groups = np.zeros((nb_blocks,), dtype=int)
    for cnt, s in enumerate(group_cdf):
        groups[group_probs >= s] = cnt

    groups = [group_names[s] for s in groups]
    return groups


def score_grouping(block_stats, groups: List[Group]):
    # compute global probs
    # we want the individual groups to have stats as close as possible to the global stats
    global_stats = np.mean(block_stats, axis=0)
    # exclude groups with zero occurrence from score
    non_zero_stats = np.where(global_stats > 0)[0]

    # score each group
    group_stats = groupstats(block_stats, groups)
    group_scores = {}
    for group_name, group_stat in group_stats.items():
        tmp = np.log2((group_stat[non_zero_stats] + 0.0000001) / global_stats[non_zero_stats])
        group_scores[group_name] = np.sum(np.abs(tmp))

    # compute total score over all groups
    total_score = np.sum(list(group_scores.values()))
    return total_score


def opt_grouping(
    block_stats: np.ndarray, group_sizes: List[Group], group_names: List[Group], nb_perms: int = 100
) -> List[Group]:
    nb_blocks = len(block_stats)
    grouping = group_blocks(nb_blocks, group_sizes, group_names)
    grouping = np.array(grouping)

    best_perm = []
    best_score = np.inf
    for _ in range(nb_perms):
        perm = np.random.permutation(grouping)
        score = score_grouping(block_stats, grouping)

        if score < best_score:
            best_score = score
            best_perm = perm

    return best_perm


def block_data(data: np.ndarray, block_size: int):
    split_points = np.array(range(0, data.shape[0], block_size))
    split_points[-1] = data.shape[0]
    # Returns a view so free in terms of memory.
    blocks = np.array_split(data, split_points[1:])[:-1]

    # ALTERNATIVE: block_size will be chosen so that all blocks have roughly equal size
    # blocks = np.array_split(y, np.floor(data.shape[0] / block_size))

    # DEBUG CODE
    # print([block.shape for block in blocks])
    # print(len(blocks), blocks[1].shape, blocks[-1].shape, np.sum([len(b) for b in blocks]))
    return blocks, split_points


def blocks_from_split_points(data: np.ndarray, split_points: List[Tuple[int, int]]):
    blocks = []
    for split_point in split_points:
        blocks.append(data[split_point[0] : split_point[1]])
    return blocks


def blockstats_from_data(data: np.ndarray, block_size: int, gap: int = 0) -> Dict[Tuple[int, int], np.ndarray]:
    """_summary_

    Last block will contain overhang so is typically longer

    Args:
        data (np.ndarray): _description_
        block_size (int): _description_
        gap (int, optional)

    Returns:
        Dict[Tuple[int, int], np.array]: _description_
    """
    blocks, split_points = block_data(data, block_size)

    blockstats = {}
    for split_start, split_end, block in zip(split_points[:-1], split_points[1:], blocks):
        blockstats[(split_start, split_end)] = np.mean(block[gap : block.shape[0] - gap], axis=0)
    return blockstats


def blockstats_from_files(file_bases: List[str], class_names: Optional[List] = None) -> Dict[str, np.ndarray]:
    """_summary_

    Args:
        file_bases (List[str]): List of annotation files.
        class_names (Optional[List]): List of class_names. Defaults to None (will infer from all annotations).

    Returns:
        Dict[str, np.ndarray]: Block stats for each file name.
    """
    if class_names is None:
        class_names = []
        for file_base in file_bases:
            df = pd.read_csv(file_base)
            df = df.dropna()
            class_names.extend(list(set(df["name"])))
        class_names = list(set(class_names))

    stratify_by_number = dict()
    for file_base in file_bases:
        df = pd.read_csv(file_base)
        df = df.dropna()
        # stratify_by_number[file_base] = {n: df["name"].tolist().count(n) for n in class_names}
        stratify_by_number[file_base] = [df["name"].tolist().count(n) for n in class_names]

    return stratify_by_number


def format_by_group(groups, block_names, group_names):
    blocks = {group_name: [] for group_name in group_names}
    for block_name, group in zip(block_names, groups):
        blocks[group].append(block_name)
    return blocks


def format_by_block(blocks, block_names):
    block_dict = dict()
    for block_name, block in zip(block_names, blocks):
        block_dict[block_name] = block
    return block_dict


def block(
    block_names: List[np.ndarray],
    group_sizes: List[float],
    group_names: List[Group],
    block_stats: Optional[List[Block]] = None,
    shuffle: bool = True,
    seed: Optional[float] = None,
) -> Dict[Group, List[Block]]:
    if seed is not None:
        np.random.seed(seed)

    if block_stats is None:
        blocks = group_blocks(len(block_names), group_sizes, group_names)
        if shuffle:
            blocks = np.random.permutation(blocks)
    else:
        block_stats = np.array(list(block_stats))
        blocks = opt_grouping(block_stats, group_sizes, group_names)

    blocks = format_by_block(blocks, block_names)

    return blocks
