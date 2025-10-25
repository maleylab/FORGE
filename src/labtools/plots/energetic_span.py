from __future__ import annotations
import matplotlib.pyplot as plt

def plot_span(labels, energies):
    '''Make a simple bar plot for energetic span values.'''
    fig = plt.figure()
    plt.bar(labels, energies)
    plt.ylabel("ΔE (kcal/mol)")
    plt.title("Energetic Span")
    plt.tight_layout()
    return fig
