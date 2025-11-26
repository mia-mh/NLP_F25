#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.
from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    # class attribute that indicates that constructor needs extra args
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)

        self.A = torch.zeros(self.k, self.k, device=self.E.device)
        self.B = torch.zeros(self.k, self.V, device=self.E.device)

    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a b-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """
        # Creating necessary variables
        k = self.k
        V = self.V
        dimi = self.rnn_dim
        e = self.e
        device = self.E.device
        dtype = self.E.dtype
        # Tagging one-hot embeddings
        self.register_buffer("tag_eye", torch.eye(k, dtype=dtype, device=device))
        # Creating RNN matrices
        self.fwd = nn.Parameter(torch.empty(dimi, 1+dimi+e, dtype=dtype, device=device))
        self.bwd = nn.Parameter(torch.empty(dimi, 1+e+dimi, dtype=dtype, device=device))
        # Creating feature networks and final linear layer weights
        self.UA = nn.Parameter(torch.empty(dimi, 1+dimi+k+k+dimi, dtype=dtype, device=device))
        self.UB = nn.Parameter(torch.empty(dimi, 1+dimi+k+e+dimi, dtype=dtype, device=device))
        self.theta_A = nn.Parameter(torch.empty(dimi, dtype=dtype, device=device))
        self.theta_B = nn.Parameter(torch.empty(dimi, dtype=dtype, device=device))
        # Using xavier uniform initialization for the matrices and normal initialization for the vectors
        nn.init.xavier_uniform_(self.fwd)
        nn.init.xavier_uniform_(self.bwd)
        nn.init.xavier_uniform_(self.UA)
        nn.init.xavier_uniform_(self.UB)
        nn.init.normal_(self.theta_A, mean=0.0, std=1.0)
        nn.init.normal_(self.theta_B, mean=0.0, std=1.0)
        # Dummy A and B to satisfy parent CRF code
        

        
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)   
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""
        # Creating necessary variables and handling degenerate case when dimensionality is 0
        W = len(isent)
        dimi = self.rnn_dim
        device = self.E.device
        dtype = self.E.dtype
        if dimi == 0:
            self.h_fwd = torch.zeros(W, 0, dtype=dtype, device=device)
            self.h_bwd = torch.zeros(W, 0, dtype=dtype, device=device)
            return
        # Extracting all word IDs and looking up all embeddings at once
        # IntegerizedSentence items are (wordID, tagID)
        word_ids = torch.tensor([w for (w, _) in isent], 
                                dtype=torch.long, device=device)

        # Mark OOV words as -1
        word_ids[ (word_ids < 0) | (word_ids >= self.E.size(0)) ] = -1
        valid_mask = word_ids >= 0
        word_embeddings = torch.zeros(W, self.e, dtype=dtype, device=device)
        if valid_mask.any():
            word_embeddings[valid_mask] = self.E[word_ids[valid_mask]]
        # Performing forward RNN
        self.h_fwd = torch.zeros(W, dimi, dtype=dtype, device=device)
        h_old = torch.zeros(dimi, dtype=dtype, device=device)
        ones = torch.ones(1, dtype=dtype, device=device)
        
        for i in range(W):
            w_vec = word_embeddings[i]
            x = torch.cat([ones, h_old, w_vec], dim=0)
            h = torch.sigmoid(self.fwd @ x)
            self.h_fwd[i] = h
            h_old = h
        # Performing backward RNN
        self.h_bwd = torch.zeros(W, dimi, dtype=dtype, device=device)
        h_new = torch.zeros(dimi, dtype=dtype, device=device)
        
        for i in range(W - 1, -1, -1):
            w_vec = word_embeddings[i]
            x = torch.cat([ones, w_vec, h_new], dim=0)
            h = torch.sigmoid(self.bwd @ x)
            self.h_bwd[i] = h
            h_new = h

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""
        # Creating necessary variables and handling degenerate case when dimensionality is 0
        W = len(sentence)
        k = self.k
        dimi = self.rnn_dim
        device = self.E.device
        dtype = self.E.dtype
        if dimi == 0:
            return torch.ones(k, k, dtype=dtype, device=device)
        if position - 1 >= 0:
            h_l = self.h_fwd[position - 1]
        else:
            h_l = torch.zeros(dimi, dtype=dtype, device=device)
        if 0 <= position < W:
            h_r = self.h_bwd[position]
        else:
            h_r = torch.zeros(dimi, dtype=dtype, device=device)
        # Building feature vectors for all (s,t) pairs and tag embeddings, and concatenating along feature dimension
        one = torch.ones(1, dtype=dtype, device=device)
        tag_eye = self.tag_eye.to(device=device, dtype=dtype)
        s_1h = tag_eye.unsqueeze(1).repeat(1, k, 1)   # (k × k × k)
        t_1h = tag_eye.unsqueeze(0).repeat(k, 1, 1)   # (k × k × k)

        ones = one.view(1, 1, 1).expand(k, k, 1)
        h_left = h_l.view(1, 1, dimi).expand(k, k, dimi)
        h_right = h_r.view(1, 1, dimi).expand(k, k, dimi)
        x = torch.cat([ones, h_left, s_1h, t_1h, h_right], dim=2)
        x = x.view(k * k, 1+dimi+k+k+dimi)
        # Calculating potentials
        h = torch.sigmoid(self.UA @ x.t()).t()
        scs = h @ self.theta_A
        scores = scs.view(k, k)
        phi_A = torch.exp(scores)
        return phi_A
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""
        # Creating necessary variables, initializing phi_B, and getting word ids at current position
        k = self.k
        dimi = self.rnn_dim
        V = self.V
        W = len(sentence)
        device = self.E.device
        dtype = self.E.dtype
        phi_B = torch.ones(k, V, dtype=dtype, device=device)
        elt = sentence[position]
        if len(elt) == 2:
            wid, _ = elt
        else:
            _, wid, _ = elt
        #  Handling oov words
        if not (0 <= wid < V):
            return phi_B     
        if not (0 <= wid < self.E.size(0)):
            w_vec = torch.zeros(self.e, dtype=dtype, device=device)
        else:
            w_vec = self.E[wid]
        # Handling degenerate case when dimensionality is 0
        if dimi == 0:
            return phi_B
        if position - 1 >= 0:
            h_l = self.h_fwd[position - 1]
        else:
            h_l = torch.zeros(dimi, dtype=dtype, device=device)
        if 0 <= position < W:
            h_r = self.h_bwd[position]
        else:
            h_r = torch.zeros(dimi, dtype=dtype, device=device)
        w_vec = self.E[wid]
        # Building feature vectors for all tags
        one = torch.ones(1, dtype=dtype, device=device)
        tag_eye = self.tag_eye.to(device=device, dtype=dtype)
        ones = one.view(1, 1).expand(k, 1)
        h_left = h_l.view(1, dimi).expand(k, dimi)
        h_right = h_r.view(1, dimi).expand(k, dimi)
        w_rep = w_vec.view(1, self.e).expand(k, self.e)
        x = torch.cat([ones, h_left, tag_eye, w_rep, h_right], dim=1)
        assert x.size(1) == 1+dimi+k+self.e+dimi
        # Calculating potentials
        h = torch.sigmoid(self.UB @ x.t()).t()
        scores = h @ self.theta_B
        phi_B[:, wid] = torch.exp(scores)
        return phi_B