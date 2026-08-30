"""Matplotlib figure builders shared across Streamlit pages."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def correlation_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def metric_bar_chart(labels: list[str], values: list[float], ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def confusion_matrix_plot(matrix: np.ndarray, labels: list):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def scatter_projection(projection: np.ndarray, labels: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    return fig
