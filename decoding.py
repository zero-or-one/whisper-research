from whisper.decoding import *



@dataclass(frozen=True)
class DecodingOptions(DecodingOptions):
    # if true and beam search is used, return beam_size best results
    return_candidates: bool = False 



class DecodingTask(DecodingTask):
    def __init__(self, model: "Whisper", options: DecodingOptions):
        super().__init__(model, options)
        self.return_candidates = options.return_candidates if hasattr(options, 'return_candidates') else False


    @torch.no_grad()
    def run(self, mel: Tensor) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)
        
        # MODIFICATION: return all beam_size best results
        if self.return_candidates:
            # We assume that tokens.shape[0] = 1, not barch inference
            tokens_seq = tokens[0]
            texts = []
            sum_logprobs_seq = sum_logprobs[0]  # Get logprobs for the first (and only) audio sample
            avg_logprobs = []

            for tokens, seq_logprobs in zip(tokens_seq, sum_logprobs_seq):
                # remove the initial tokens
                tokens = tokens[self.sample_begin:]
                texts.append(tokenizer.decode(tokens).strip())
                # Calculate average logprob: sum of logprobs divided by sequence length (including EOS token)
                avg_logprob = seq_logprobs / (len(tokens) + 1)
                avg_logprobs.append(avg_logprob)

            # convert tokens to list
            tokens_seq = [t.tolist() for t in tokens_seq]

            return [
                DecodingResult(
                    audio_features=audio_features,
                    language=languages[0],
                    tokens=tokens,
                    text=text,
                    avg_logprob=avg_logprob,  # Now using the calculated average logprob
                    no_speech_prob=no_speech_probs[0],
                    temperature=self.options.temperature,
                    compression_ratio=compression_ratio(text),
                )
                for text, tokens, avg_logprob in zip(texts, tokens_seq, avg_logprobs)
            ]

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)
    
    # MODIFICATION: return n best results
    if hasattr(options, 'return_candidates') and options.return_candidates:
        return results

    return result[0] if single else result