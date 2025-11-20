""" # ======================================
#   BASIC STATISTICS 
# ======================================

from collections import Counter
import math

def mean(data):
    return sum(data) / len(data)

def median(data):
    d = sorted(data)
    n = len(d)
    mid = n // 2
    if n % 2 == 1:
        return d[mid]
    else:
        return (d[mid - 1] + d[mid]) / 2

def mode(data):
    counts = Counter(data)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    if len(modes) == 1:
        return modes[0]
    return modes  # multimodal

def variance(data, sample=True):
    m = mean(data)
    n = len(data)
    if sample and n > 1:
        return sum((x - m) ** 2 for x in data) / (n - 1)
    else:
        return sum((x - m) ** 2 for x in data) / n

def std_dev(data, sample=True):
    return math.sqrt(variance(data, sample))

def data_range(data):
    return max(data) - min(data)

def quartiles(data):
    d = sorted(data)
    n = len(d)
    mid = n // 2
    
    if n % 2 == 0:
        lower = d[:mid]
        upper = d[mid:]
    else:
        lower = d[:mid]
        upper = d[mid+1:]
    
    Q1 = median(lower)
    Q2 = median(d)
    Q3 = median(upper)
    return Q1, Q2, Q3

def iqr(data):
    Q1, _, Q3 = quartiles(data)
    return Q3 - Q1

def summary(data):
    print("===== BASIC STATISTICS SUMMARY =====")
    print(f"Count: {len(data)}")
    print(f"Mean: {mean(data)}")
    print(f"Median: {median(data)}")
    print(f"Mode: {mode(data)}")
    print(f"Variance: {variance(data):.4f}")
    print(f"Standard Deviation: {std_dev(data):.4f}")
    print(f"Range: {data_range(data)}")
    Q1, Q2, Q3 = quartiles(data)
    print(f"Q1: {Q1}, Q2 (Median): {Q2}, Q3: {Q3}")
    print(f"IQR: {iqr(data)}")
    print("====================================")


# =============================
# Example Usage
# =============================


dataset = [12, 15, 11, 13, 12, 10, 17, 15, 12]
summary(dataset)
 """


# ===========================================
#   BASIC STATISTICS ALL-IN-ONE (WITH LIBS)
# ===========================================

import numpy as np
import pandas as pd
from scipy import stats

def basic_stats(data):
    data = np.array(data)

    stats_dict = {
        "count": len(data),
        "mean": np.mean(data),
        "median": np.median(data),
        "mode": stats.mode(data, keepdims=True).mode[0],
        "variance": np.var(data, ddof=1),
        "std_dev": np.std(data, ddof=1),
        "min": np.min(data),
        "max": np.max(data),
        "range": np.ptp(data),
        "Q1": np.percentile(data, 25),
        "Q2": np.percentile(data, 50),
        "Q3": np.percentile(data, 75),
        "IQR": np.percentile(data, 75) - np.percentile(data, 25),
        "skewness": stats.skew(data),
        "kurtosis": stats.kurtosis(data)
    }

    return stats_dict


def print_summary(data):
    stats_dict = basic_stats(data)

    print("========== BASIC STATISTICS SUMMARY ==========")
    for k, v in stats_dict.items():
        print(f"{k:10}: {v}")
    print("==============================================")


# ================================
# Example usage
# ================================

if __name__ == "__main__":
    dataset = [12, 15, 11, 13, 12, 10, 17, 15, 12]
    print_summary(dataset)
