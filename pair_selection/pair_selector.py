import numpy as np
import pandas as pd
from itertools import combinations
from .utils import pair_to_rank_map
from scipy.stats import spearmanr
from typing import Callable, List, Tuple

class PairSelector:
    def __init__(self, W: int = 10, train_frac: float = 0.7, top_n: int = 50):
        """
        Initialize the pair selection framework.

        Args:
            W (int): Lookback window length in days.
            train_frac (float): Fraction of window to use as training set.
            top_k (int): Number of top pairs to return.
        """
        self.W = W
        self.train_frac = train_frac
        self.top_n = top_n

    def get_lookback(self, df: pd.DataFrame, T0: pd.Timestamp) -> pd.DataFrame:
        """
        Extracts the lookback window from [T0 - W, T0). 

        Args:
            df: Full price data (datetime index, coin symbols as columns).
            T0: Current time point for selection.

        Returns:
            A subset of df from [T0 - W days, T0).
        """
        return df[(df.index >= T0 - pd.Timedelta(days=self.W)) & (df.index < T0)]

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a time window into training and validation segments.

        Args:
            df: Data from lookback window.

        Returns:
            Tuple of (training_data, validation_data).
        """
        idx = int(len(df) * self.train_frac)
        return df.iloc[:idx], df.iloc[idx:]

    def compute_profitability_scores(self, df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        """
        Scores each pair on the validation set by computing a crude profitability proxy based on a heuristic of the spread.
        Currently, the heuristic is a functino of the amplitude and number of zero crossings of the spread (as a gauge of mean-reversion tendency). 

        Args:
            df: Validation set.
            pairs: List of asset pairs (tuples of symbols).

        Returns:
            List of (asset A, asset B, profitability score), sorted descending.
        """
        scores = []
        for a, b in pairs:
            spread = df[a] - df[b]
            amp = spread.max() - spread.min()  # Amplitude
            zero_cross = ((spread.shift(1) * spread) < 0).sum()  # # of zero crossings
            score = amp + 0.1 * zero_cross  # Composite profitability proxy
            scores.append((a, b, score))
        return sorted(scores, key=lambda x: -x[2])  # Descending score = better

    def rank_by_metrics(
        self, df: pd.DataFrame, pairs: List[Tuple[str, str]], metric_fns: List[Callable[[pd.Series, pd.Series], float]]
    ) -> List[List[Tuple[str, str, float]]]:
        """
        Applies each distance metric to training data and ranks the pairs.

        Args:
            df: Training set.
            pairs: List of asset pairs.
            metric_fns: List of distance functions, each taking two Series and returning a float.

        Returns:
            A list of ranked lists, one per metric. Each inner list is [(a, b, distance), ...], sorted by increasing distance.
        """
        all_rankings = []
        for fn in metric_fns:
            distances = [(a, b, fn(df[a], df[b])) for a, b in pairs]
            ranked = sorted(distances, key=lambda x: x[2])  # Smaller distance = better
            all_rankings.append(ranked)
        return all_rankings

    def ranking_losses(
        self, metric_rankings: List[List[Tuple[str, str, float]]], val_ranking: List[Tuple[str, str, float]]
    ) -> List[float]:
        """
        Computes how well each metric's ranking agrees with validation profitability ranking.

        Args:
            metric_rankings: List of metric-based ranked pair lists.
            val_ranking: Ground truth validation ranking from profitability.

        Returns:
            List of loss values (1 - Spearman correlation), one per metric.
        """
        true_map = pair_to_rank_map(val_ranking)
        losses = []
        for ranking in metric_rankings:
            pred_map = pair_to_rank_map(ranking)
            shared = list(set(pred_map) & set(true_map))
            pred = [pred_map[p] for p in shared]
            true = [true_map[p] for p in shared]
            losses.append(1 - spearmanr(pred, true).correlation)  
        return losses

    def assign_weights(self, losses: List[float]) -> np.ndarray:
        """
        Assigns weights to distance metrics based on performance (lower loss = higher weight).

        Args:
            losses: List of ranking losses for each metric.

        Returns:
            Normalized array of weights summing to 1.
        """
        idx = np.argsort(losses)
        w = np.zeros(len(losses))
        for i, ix in enumerate(idx):
            w[ix] = 1 / (i + 2)  # Best = 1/2, next = 1/3, etc.
        return w / w.sum()

    def aggregate(
        self, metric_rankings: List[List[Tuple[str, str, float]]], weights: np.ndarray
    ) -> List[Tuple[str, str, float]]:
        """
        Aggregates the rankings across all metrics using weighted average of ranks.

        Args:
            metric_rankings: Ranked lists from each metric.
            weights: Normalized weights for each metric.

        Returns:
            Final ranked list of pairs (a, b, score), sorted by ascending weighted rank.
        """
        rank_maps = [pair_to_rank_map(r) for r in metric_rankings]
        pairs = rank_maps[0].keys()
        scores = []
        for pair in pairs:
            weighted_rank = sum(w * rm[pair] for w, rm in zip(weights, rank_maps))
            scores.append((pair[0], pair[1], weighted_rank))
        return sorted(scores, key=lambda x: x[2])

    def select(
        self,
        df: pd.DataFrame,
        T0: pd.Timestamp,
        metric_fns: List[Callable[[pd.Series, pd.Series], float]]
    ) -> List[Tuple[str, str, float]]:
        """
        Master pipeline: selects top pairs at time T0 using weighted metric aggregation.

        Args:
            df: Full price history (datetime index, coin columns).
            T0: Rebalance time (anchor point).
            metric_fns: List of distance metric functions.

        Returns:
            List of top_k selected pairs: (a, b, score).
        """
        # 1. Extract window ending at T0
        window = self.get_lookback(df, T0)

        # 2. Split into training and validation sets
        train, val = self.split(window)

        # 3. Generate all asset pairs
        pairs = list(combinations(train.columns, 2))

        # 4. Compute profitability-based ranking (validation set)
        val_rank = self.compute_profitability_scores(val, pairs)

        # 5. Apply distance metrics to training set
        metric_ranks = self.rank_by_metrics(train, pairs, metric_fns)

        # 6. Evaluate each ranking vs. validation ranking
        losses = self.ranking_losses(metric_ranks, val_rank)

        # 7. Assign weights to metrics (better = higher weight)
        weights = self.assign_weights(losses)

        # 8. Aggregate rankings using metric weights
        final_rank = self.aggregate(metric_ranks, weights)

        # 9. Return top K pairs
        return final_rank[:self.top_k]
