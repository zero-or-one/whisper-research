from whisper.timing import *
import pickle
import os


def find_alignment(
    model: "Whisper",
    tokenizer: Tokenizer,
    text_tokens: List[int],
    mel: torch.Tensor,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
    save_attention: bool = True,
) -> Tuple[List[WordTiming], Optional[Dict[str, torch.Tensor]]]:
    """
    Find alignment between text tokens and audio frames.
    
    Args:
        save_attention: If True, returns attention patterns along with word timings
    
    Returns:
        Tuple of (word_timings, attention_dict) where attention_dict is None if save_attention=False
    """
    if len(text_tokens) == 0:
        return [], None
        
    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    ).to(model.device)

    # Install hooks on the cross attention layers to retrieve the attention weights
    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1][0])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    from .model import disable_sdpa
    with torch.no_grad():#, disable_sdpa():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]
        sampled_logits = logits[len(tokenizer.sot_sequence):, :tokenizer.eot]
        token_probs = sampled_logits.softmax(dim=-1)
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens]
        text_token_probs = text_token_probs.tolist()

    for hook in hooks:
        hook.remove()

    # heads * tokens * frames
    weights = torch.stack([QKs[l][h] for l, h in model.alignment_heads.indices().T])
    weights = weights[:, :, :num_frames // 2]
    weights = (weights * qk_scale).softmax(dim=-1)

    # Add all weights to the attention_dict
    all_weights = torch.stack([QKs[l][h] for l in range(len(QKs)) for h in range(QKs[l].shape[0])])
    all_weights = all_weights[:, :, :num_frames // 2]
    all_weights = (all_weights * qk_scale).softmax(dim=-1)    
    
    # Store original weights if requested
    attention_dict = None
    if save_attention:
        attention_dict = {
            'all_raw_weights': all_weights.clone(),  # Attention weights before normalization
            'raw_weights': weights.clone(),  # Original attention weights
            'alignment_heads': model.alignment_heads.indices().T.clone(),  # Which heads were used
            'token_probs': torch.tensor(text_token_probs),  # Token probabilities
        }
    
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = median_filter(weights, medfilt_width)
    matrix = weights.mean(axis=0)
    matrix = matrix[len(tokenizer.sot_sequence):-1]
    
    if save_attention:
        attention_dict.update({
            'normalized_weights': weights.clone(),  # After normalization
            'final_matrix': matrix.clone(),  # Final attention matrix used for alignment
        })

    text_indices, time_indices = dtw(-matrix)
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])

    if len(word_tokens) <= 1:
        return [], attention_dict

    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    word_timings = [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]
    return word_timings, attention_dict


def add_word_timestamps(
    *,
    segments: List[dict],
    model: "Whisper",
    tokenizer: Tokenizer,
    mel: torch.Tensor,
    num_frames: int,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    last_speech_timestamp: float,
    **kwargs,
):
    if len(segments) == 0:
        return

    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot]
        for segment in segments
    ]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    attention_dict = None
    alignment, attention_dict = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)

    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]
    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2

    # hack: truncate long words at sentence boundaries.
    # a better segmentation algorithm based on VAD should be able to replace this.
    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        # ensure words at sentence boundaries are not longer than twice the median word duration.
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks:
                    alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + timing.start, 2),
                        end=round(time_offset + timing.end, 2),
                        probability=timing.probability,
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # hack: truncate long words at segment boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(words) > 0:
            # ensure the first and second word after a pause is not longer than
            # twice the median word duration.
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (
                    len(words) > 1
                    and words[1]["end"] - words[0]["start"] > max_duration * 2
                )
            ):
                if (
                    len(words) > 1
                    and words[1]["end"] - words[1]["start"] > max_duration
                ):
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            # prefer the segment-level start timestamp if the first word is too long.
            if (
                segment["start"] < words[0]["end"]
                and segment["start"] - 0.5 > words[0]["start"]
            ):
                words[0]["start"] = max(
                    0, min(words[0]["end"] - median_duration, segment["start"])
                )
            else:
                segment["start"] = words[0]["start"]

            # prefer the segment-level end timestamp if the last word is too long.
            if (
                segment["end"] > words[-1]["start"]
                and segment["end"] + 0.5 < words[-1]["end"]
            ):
                words[-1]["end"] = max(
                    words[-1]["start"] + median_duration, segment["end"]
                )
            else:
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words

        # MODIFICATION: save the dictionary for alignment visualization
        os.mkdir('results', exists_ok=True)
        with open('results/attention_dict.pkl', 'wb') as f:
            pickle.dump(attention_dict, f)