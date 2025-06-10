def pair_to_rank_map(ranked_list):
    """
    Converts a ranked list of (asset A, asset B, score) tuples into a dictionary
    mapping each (A, B) pair to its rank index.

    Args:
        ranked_list (List[Tuple[str, str, float]]): 
            A list of ranked asset pairs, sorted from best to worst (e.g., lowest distance or highest profitability).

    Returns:
        Dict[Tuple[str, str], int]: 
            A mapping from each (A, B) pair to its position (0-based rank) in the list.
    """
    return {(a, b): i for i, (a, b, _) in enumerate(ranked_list)}
