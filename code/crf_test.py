#!/usr/bin/env python3

# Subclass ConditionalRandomFieldBackprop to get a model that uses some
# contextual features of your choice.  This lets you test the revision to hmm.py
# that uses those features.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Tag, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldTest(ConditionalRandomFieldBackprop):
    """A CRF with some arbitrary non-stationary features, for testing."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        # an __init__() call to the nn.Module class must be made before assignment on the child.
        nn.Module.__init__(self)  

        self.E = lexicon          # rows are word embeddings
        self.e = lexicon.size(1)  # dimensionality of word embeddings
        self.rnn_dim = rnn_dim

        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        # k = number of tags, V = vocab size (including EOS/BOS in CRF code)
        self.k = len(self.tagset)
        self.V = len(self.vocab)

        # BiLSTM over embeddings
        self.rnn = nn.LSTM(
            input_size=self.e,
            hidden_size=self.rnn_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        hidden_dim = 2 * self.rnn_dim

        # Transition scorer: from context vector -> k*k transition potentials
        self.transition_layer = nn.Linear(hidden_dim, self.k * self.k)

        # Emission/state scorer: from context vector -> k state potentials
        # (we’ll put these into column w_j of B_at)
        self.emission_layer = nn.Linear(hidden_dim, self.k)

        # Initialize small/random like in hmm.py
        for m in (self.transition_layer, self.emission_layer):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            nn.init.zeros_(m.bias)

        # Optional: create dummy A/B so that any code that expects them exists,
        # even though we never use them in this subclass.
        self.A = torch.zeros(self.k, self.k)
        self.B = torch.zeros(self.k, self.V)

        # Cache for sentence-level RNN outputs
        self._cached_sentence = None
        self._cached_hiddens: Tensor | None = None


    @override
    def updateAB(self) -> None:
        # Your non-stationary A_at() and B_at() might not make any use of the
        # stationary A and B matrices computed by the parent.  So we override
        # the parent so that we won't waste time computing self.A, self.B.
        #
        # But if you decide that you want A_at() and B() at to refer to self.A
        # and self.B (for example, multiplying stationary and non-stationary
        # potentials), then you'll still need to compute them; in that case,
        # don't override the parent in this way.
        pass   # do nothing

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        """
        Return transition potentials φ_j(s -> t) as a (k x k) matrix for 
        the transition into position `position`.

        We use the BiLSTM hidden state at position-1 as context.
        """
        h_seq = self._compute_hiddens(sentence)
        device = h_seq.device

        # For position 0 there is no real transition; use a neutral matrix
        if position == 0:
            A = torch.zeros(self.k, self.k, device=device)
            A[:, self.bos_t] = 0.0
            A[self.eos_t, :] = 0.0
            return A

        # Context from previous position
        h_prev = h_seq[position - 1]           # (2*rnn_dim,)
        scores = self.transition_layer(h_prev) # (k*k,)
        scores = scores.view(self.k, self.k)   # (k, k)

        # Exponentiate to get positive potentials (CRF style, unnormalized)
        A = torch.exp(scores)

        # Structural zeros: never transition INTO BOS, or OUT OF EOS
        A[:, self.bos_t] = 0.0
        A[self.eos_t, :] = 0.0

        return A

        
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """
        Return emission/state potentials as a (k x V) matrix for position `position`.

        Only the column of the actually observed word w_j is filled with meaningful
        potentials; other columns are left at 0 (they won't be used).
        """
        h_seq = self._compute_hiddens(sentence)
        device = h_seq.device

        w_j, _ = sentence[position]

        # Context at the current position
        h = h_seq[position]                      # (2*rnn_dim,)
        tag_scores = self.emission_layer(h)      # (k,)
        tag_potentials = torch.exp(tag_scores)   # positive, unnormalized

        B = torch.zeros(self.k, self.V, device=device)

        # Only real words (ignore BOS/EOS "words" if they’re outside range)
        if 0 <= w_j < self.V:
            # For each tag t, the potential φ_j(t) sits at B[t, w_j]
            B[:, w_j] = tag_potentials

        # Structural zeros: BOS and EOS tags never emit real words
        B[self.eos_t, :] = 0.0
        B[self.bos_t, :] = 0.0

        return B

    
    def _compute_hiddens(self, sentence) -> Tensor:
        """Run BiLSTM over the sentence’s word embeddings and cache the result.

        Returns a tensor of shape (len(sentence), 2*rnn_dim).
        """
        # Simple identity-based cache: assumes the same list object is reused
        if self._cached_sentence is sentence and self._cached_hiddens is not None:
            return self._cached_hiddens

        # Integerized sentence is a list of (word_index, tag_index_or_None)
        word_indices = [w for (w, _) in sentence]

        # Look up embeddings: (n, e) -> add batch dim -> (1, n, e)
        X = self.E[word_indices].unsqueeze(0)

        # Make sure everything is on the same device as the model parameters
        device = next(self.parameters()).device
        X = X.to(device)
        self.rnn.to(device)

        h_seq, _ = self.rnn(X)    # (1, n, 2*rnn_dim)
        h_seq = h_seq.squeeze(0)  # (n, 2*rnn_dim)

        self._cached_sentence = sentence
        self._cached_hiddens = h_seq
        return h_seq
