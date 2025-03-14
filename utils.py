import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager, rc



def plot_attention_with_spectrogram(attention_matrix, mel_spectrogram, tokens, figsize=(15, 10), block=-1, head=-1):
    """
    Create a visualization combining attention matrix, mel spectrogram, and tokens.
    Parameters:
    - attention_matrix: numpy array of shape (encoder_steps, decoder_steps)
    - mel_spectrogram: numpy array of shape (n_mels, encoder_steps)
    - tokens: list of decoded tokens
    - figsize: tuple for figure size
    """
    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])

    # Plot mel spectrogram on the left
    ax_mel = fig.add_subplot(gs[0, 0])
    mel_db = mel_spectrogram
    im = ax_mel.imshow(mel_db, aspect='auto', cmap='magma', origin='lower')
    ax_mel.set_ylabel('Mel Frequency')
    ax_mel.set_xticks([])

    # Plot attention matrix in the center
    ax_attention = fig.add_subplot(gs[0, 1])
    attention_map = ax_attention.imshow(attention_matrix.T,
                                        aspect='auto',
                                        origin='lower',
                                        cmap='Blues',
                                        interpolation='nearest')
    ax_attention.set_xticks([])
    ax_attention.set_yticks([])

    # Add colorbar
    plt.colorbar(attention_map, ax=ax_attention, label='Attention Weight')

    # Plot tokens at the bottom
    ax_tokens = fig.add_subplot(gs[1, 1])
    decoder_steps = attention_matrix.shape[1]
    token_positions = np.arange(len(tokens))*5.5+1.7  # Directly map tokens to indices
    ax_tokens.set_ylim(-1, 1)
    ax_tokens.set_xlim(-0.5, decoder_steps - 0.5)

    # Add tokens with vertical lines
    for idx, (pos, token) in enumerate(zip(token_positions, tokens)):
        ax_tokens.axvline(x=pos, color='gray', linestyle=':', alpha=0.5)
        ax_tokens.text(pos, 0, token, ha='center', va='center', fontsize=10)

    # Remove token axis spines and ticks
    ax_tokens.set_xticks([])
    ax_tokens.set_yticks([])
    for spine in ax_tokens.spines.values():
        spine.set_visible(False)

    # Empty plot for grid alignment
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.set_xticks([])
    ax_empty.set_yticks([])
    for spine in ax_empty.spines.values():
        spine.set_visible(False)

    # Add titles
    ax_mel.set_title('Mel Spectrogram')
    ax_attention.set_title(f'Attention Matrix (Block {block}, Head {head})')
    plt.tight_layout()
    return fig
