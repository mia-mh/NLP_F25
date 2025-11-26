#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
import os, time
from typing import Callable, List, Optional, cast
from typeguard import typechecked

import torch
from torch import Tensor, cuda, nn
from jaxtyping import Float

from tqdm import tqdm # type: ignore
import pickle

from integerize import Integerizer
from corpus import BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, IntegerizedSentence, Word

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# HMM tagger
###
class HiddenMarkovModel:
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """
    
    # As usual in Python, attributes and methods starting with _ are intended as private;
    # in this case, they might go away if you changed the parametrization of the model.

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The `unigram` flag
        says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended to
        support higher-order HMMs: trigram HMMs used to be popular.)"""

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        if vocab[-2:] != [EOS_WORD, BOS_WORD]:
            raise ValueError("final two types of vocab should be EOS_WORD, BOS_WORD")

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.unigram = unigram     # do we fall back to a unigram model?

        self.tagset = tagset
        self.vocab = vocab

        # Useful constants that are referenced by the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        if self.bos_t is None or self.eos_t is None:
            raise ValueError("tagset should contain both BOS_TAG and EOS_TAG")
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize model parameters
 
    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).  
        We respect structural zeroes ("Don't guess when you know").
            
        If you prefer, you may change the class to represent the parameters in logspace,
        as discussed in the reading handout as one option for avoiding underflow; then name
        the matrices lA, lB instead of A, B, and construct them by logsoftmax instead of softmax."""

        ###
        # Randomly initialize emission probabilities.
        # A row for an ordinary tag holds a distribution that sums to 1 over the columns.
        # But EOS_TAG and BOS_TAG have probability 0 of emitting any column's word
        # (instead, they have probability 1 of emitting EOS_WORD and BOS_WORD (respectively), 
        # which don't have columns in this matrix).
        ###
        WB = 0.01*torch.rand(self.k, self.V)  # choose random logits
        self.B = WB.softmax(dim=1)            # construct emission distributions p(w | t)
        self.B[self.eos_t, :] = 0             # EOS_TAG can't emit any column's word
        self.B[self.bos_t, :] = 0             # BOS_TAG can't emit any column's word
        
        ###
        # Randomly initialize transition probabilities, in a similar way.
        # Again, we respect the structural zeros of the model.
        ###
        rows = 1 if self.unigram else self.k
        WA = 0.01*torch.rand(rows, self.k)
        WA[:, self.bos_t] = -inf    # correct the BOS_TAG column
        self.A = WA.softmax(dim=1)  # construct transition distributions p(t | s)
        if self.unigram:
            # A unigram model really only needs a vector of unigram probabilities
            # p(t), but we'll construct a bigram probability matrix p(t | s) where 
            # p(t | s) doesn't depend on s. 
            # 
            # By treating a unigram model as a special case of a bigram model,
            # we can simply use the bigram code for our unigram experiments,
            # although unfortunately that preserves the O(nk^2) runtime instead
            # of letting us speed up to O(nk) in the unigram case.
            self.A = self.A.repeat(self.k, 1)   # copy the single row k times  

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")


    def _E_step_supervised(self, isent: IntegerizedSentence, mult: float = 1.0) -> None:
        for j in range(1, len(isent)):
            w_j, t_j = isent[j]
            t_prev = isent[j-1][1]
            
            # Transition count: t_prev -> t_j
            self.A_counts[t_prev, t_j] += mult
            
            # Emission count: skip boundary pseudo-words
            if w_j < self.V:
                self.B_counts[t_j, w_j] += mult


    def E_step(self, isent: IntegerizedSentence, mult: float = 1) -> None:
        """Runs the forward backward algorithm on the given sentence. The forward step computes
        the alpha probabilities.  The backward step computes the beta probabilities and
        adds expected counts to self.A_counts and self.B_counts.  
        
        The multiplier `mult` says how many times to count this sentence. 
        
        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""

        # Dispatch based on whether sentence is fully tagged
        fully_tagged = all(t is not None for _, t in isent)
        
        if fully_tagged:
            # Supervised: just collect hard counts
            self._E_step_supervised(isent, mult)
        else:
            # Unsupervised/partially supervised: use forward-backward
            log_Z_forward = self.forward_pass(isent)
            log_Z_backward = self.backward_pass(isent, mult=mult)
            
            # Verify forward and backward give same log Z
            f_ok = torch.isfinite(log_Z_forward)
            b_ok = torch.isfinite(log_Z_backward)
            
            if f_ok and b_ok:
                if not torch.isclose(log_Z_forward, log_Z_backward, rtol=1e-3, atol=1e-4):
                    import logging
                    logging.warning(
                        f"forward/backward mismatch: forward={log_Z_forward.item():.6f}, "
                        f"backward={log_Z_backward.item():.6f}"
                    )

    def M_step(self, λ: float) -> None:
        """Set the transition and emission matrices (A, B), using the expected
        counts (A_counts, B_counts) that were accumulated by the E step.
        The `λ` parameter will be used for add-λ smoothing.
        We respect structural zeroes ("don't guess when you know")."""

        # we should have seen no emissions from BOS or EOS tags
        assert self.B_counts[self.eos_t:self.bos_t, :].any() == 0, 'Your expected emission counts ' \
                'from EOS and BOS are not all zero, meaning you\'ve accumulated them incorrectly!'

        # Update emission probabilities (self.B).
        self.B_counts += λ          # smooth the counts (EOS_WORD and BOS_WORD remain at 0 since they're not in the matrix)
        self.B = self.B_counts / self.B_counts.sum(dim=1, keepdim=True)  # normalize into prob distributions
        self.B[self.eos_t, :] = 0   # replace these nan values with structural zeroes, just as in init_params
        self.B[self.bos_t, :] = 0

        # we should have seen no "tag -> BOS" or "BOS -> tag" transitions
        assert self.A_counts[:, self.bos_t].any() == 0, 'Your expected transition counts ' \
                'to BOS are not all zero, meaning you\'ve accumulated them incorrectly!'
        assert self.A_counts[self.eos_t, :].any() == 0, 'Your expected transition counts ' \
                'from EOS are not all zero, meaning you\'ve accumulated them incorrectly!'
                
        # Update transition probabilities (self.A).  
        # Don't forget to respect the settings self.unigram and λ.
        # See the init_params() method for a discussion of self.A in the
        # unigram case.
        
                # --- YOUR CODE BELOW: compute self.A from self.A_counts ---

        # Add-λ smoothing to transition counts (do it AFTER the teacher's asserts)
        A_counts = self.A_counts + λ

        # Enforce structural zeros on the COUNTS before normalizing:
        # - no transitions INTO BOS (zero BOS column)
        # - no transitions OUT OF EOS (zero EOS row)
        A_counts[:, self.bos_t] = 0.0
        A_counts[self.eos_t, :] = 0.0

        if self.unigram:
            # Unigram HMM: estimate one p(t) row then repeat to all rows
            row = A_counts.sum(dim=0, keepdim=True)         # total arrivals to each tag t
            row[:, self.bos_t] = 0.0                        # keep BOS column structurally zero
            row_sum = row.sum(dim=1, keepdim=True) + 1e-45
            row = row / row_sum
            self.A = row.repeat(self.k, 1)
        else:
            # Bigram HMM: row-normalize to get p(t | s)
            denom = A_counts.sum(dim=1, keepdim=True) + 1e-45
            self.A = A_counts / denom

        # Guard rails after normalization (belt-and-suspenders):
        self.A[:, self.bos_t] = 0.0     # no transitions INTO BOS
        self.A[self.eos_t, :] = 0.0     # no transitions OUT OF EOS

    @typechecked
    def forward_pass(self, isent: IntegerizedSentence) -> TorchScalar:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward
        probability) as a TorchScalar.  If the sentence is not fully tagged, the 
        forward probability will marginalize over all possible tags.  
        
        As a side effect, remember the alpha probabilities and log_Z
        (store some representation of them into attributes of self)
        so that they can subsequently be used by the backward pass."""
        
        # Preallocate alpha vectors
        alpha = [torch.zeros(self.k) for _ in isent]
        alpha[0][self.bos_t] = 1.0  # One-hot at BOS
        
        log_Z_components = []
        
        # Forward loop
        for j in range(1, len(isent)):
            w_j, t_j = isent[j]
            
             # Get emission probabilities
            if w_j >= self.V:  # Boundary word (BOS_WORD or EOS_WORD)
                emission = torch.zeros(self.k)
                if w_j == self.V:  # EOS_WORD
                    emission[self.eos_t] = 1.0
                else:  # BOS_WORD
                    emission[self.bos_t] = 1.0
            else:
                emission = self.B[:, w_j]
            
            # α(j) = (α(j-1) · A) ⊙ B[:, w_j]
            alpha[j] = (alpha[j-1] @ self.A) * emission
            
            # If supervised, restrict to observed tag
            if t_j is not None:
                mask = torch.zeros(self.k)
                mask[t_j] = 1.0
                alpha[j] = alpha[j] * mask
            
            # Scaling trick (Section C.1)
            kappa_j = alpha[j].sum()
            if kappa_j > 0:
                log_Z_components.append(torch.log(kappa_j))
                alpha[j] = alpha[j] / kappa_j
            else:
                log_Z_components.append(torch.tensor(float('-inf')))
        
        # Final log Z
        final_prob = alpha[-1][self.eos_t]
        if final_prob > 0:
            log_Z = torch.log(final_prob) + sum(log_Z_components)
        else:
            log_Z = torch.tensor(float('-inf'))
        
        # Save for backward pass
        self.alpha = alpha
        self.log_Z_components = log_Z_components
        self.log_Z = log_Z
        
        return log_Z


    @typechecked
    def backward_pass(self, isent: IntegerizedSentence, mult: float = 1) -> TorchScalar:
        """Run the backwards algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  
        
        As a side effect, add the expected transition and emission counts (times 
        mult) into self.A_counts and self.B_counts.
        
        The backward pass should run over the same alpha values and log_Z 
        that the forward pass computed.  (You don't need to pass them as arguments;
        they are already stored as attributes of self.)  It should return the 
        same log_Z value that the forward pass returned.
        
        When the logging level is set to DEBUG, the beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        
        alpha = self.alpha
        log_Z_components = self.log_Z_components
        
        # Preallocate beta just as we pre-allocated alpha
        beta = [torch.zeros(self.k) for _ in isent]
        beta[-1][self.eos_t] = 1.0  # One-hot at EOS
        
        # Normalizing constant for posteriors
        Z_scaled = alpha[-1][self.eos_t]
        
        # Backward loop
        for j in range(len(isent) - 1, 0, -1):
            w_j, t_j = isent[j]
            
            # Get emission probabilities
            if w_j >= self.V:
                emission = torch.zeros(self.k)
                if w_j == self.V:
                    emission[self.eos_t] = 1.0
                else:
                    emission[self.bos_t] = 1.0
            else:
                emission = self.B[:, w_j]
            
            # Posterior at position j: γ(j) = α(j) ⊙ β(j) / Z
            gamma = (alpha[j] * beta[j]) / (Z_scaled + 1e-45)
            
            # Emission counts (Algorithm 4, line 7)
            if w_j < self.V:  # Only real words
                if t_j is None:
                    self.B_counts[:, w_j] += mult * gamma
                else:
                    self.B_counts[t_j, w_j] += mult * gamma[t_j]
            
            # Transition counts (Algorithm 4, line 11)
            kappa_j = torch.exp(log_Z_components[j-1]) if (j-1) < len(log_Z_components) else torch.tensor(1.0)
            
            xi = (
                alpha[j-1].unsqueeze(1) *  # α_s(j-1)
                self.A *                    # A[s,t]
                emission.unsqueeze(0) *     # B[t, w_j]
                beta[j].unsqueeze(0)        # β_t(j)
            ) / (kappa_j * Z_scaled + 1e-45)
            
            if t_j is None:
                self.A_counts += mult * xi
            else:
                self.A_counts[:, t_j] += mult * xi[:, t_j]
            
            # Update beta (Algorithm 4, line 12)
            beta[j-1] = self.A @ (emission * beta[j])
            if kappa_j > 0:
                beta[j-1] = beta[j-1] / kappa_j
        
        # Compute log Z from backward (for verification)
        log_Z_backward = torch.log(beta[0][self.bos_t] + 1e-45) + sum(log_Z_components)
        
        return log_Z_backward


    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        isent = self._integerize_sentence(sentence, corpus)
        
        # Work in LOG-SPACE to prevent underflow
        alpha = [torch.full((self.k,), float('-inf')) for _ in isent]
        backpointers = [torch.zeros(self.k, dtype=torch.long) for _ in isent]
        
        # Initialize: log(1) = 0 at BOS
        alpha[0][self.bos_t] = 0.0
        
        # Forward max-sum (in log space)
        for j in range(1, len(isent)):
            w_j, t_j = isent[j]
            
            # Get LOG emission probabilities
            if w_j >= self.V:
                log_emission = torch.full((self.k,), float('-inf'))
                if w_j == self.V:
                    log_emission[self.eos_t] = 0.0
                else:
                    log_emission[self.bos_t] = 0.0
            else:
                log_emission = torch.log(self.B[:, w_j] + 1e-45)
            
            # Compute LOG scores: log(α[j-1]) + log(A) + log(emission)
            log_A = torch.log(self.A + 1e-45)
            scores = alpha[j-1].unsqueeze(1) + log_A + log_emission.unsqueeze(0)
            
            # Max over previous states (in log space)
            max_scores, argmax_indices = scores.max(dim=0)
            alpha[j] = max_scores
            backpointers[j] = argmax_indices
            
            # Supervised constraint
            if t_j is not None:
                # In log space: -inf for all except observed tag
                supervised_score = alpha[j][t_j].clone()
                alpha[j][:] = float('-inf')
                alpha[j][t_j] = supervised_score
        
        # Backtrace
        tags = [0] * len(isent)
        tags[-1] = self.eos_t
        for j in range(len(isent) - 1, 0, -1):
            tags[j-1] = backpointers[j][tags[j]].item()
        
        # Map back to original sentence
        tagged_sentence = Sentence([
            (word, self.tagset[tags[j]]) 
            for j, (word, _) in enumerate(sentence)
        ])
        
        return tagged_sentence



    def _zero_counts(self):
        """Set the expected counts to 0.  
        (This creates the count attributes if they didn't exist yet.)"""
        self.A_counts = torch.zeros((self.k, self.k), requires_grad=False)
        self.B_counts = torch.zeros((self.k, self.V), requires_grad=False)

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              λ: float = 0,
              tolerance: float = 0.001,
              max_steps: int = 50000,
              save_path: Optional[Path|str] = "my_hmm.pkl") -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        We will stop when the relative improvement of the development loss,
        since the last epoch, is less than the tolerance.  In particular,
        we will stop when the improvement is negative, i.e., the development loss 
        is getting worse (overfitting).  To prevent running forever, we also
        stop if we exceed the max number of steps."""
        
        if λ < 0:
            raise ValueError(f"{λ=} but should be >= 0")
        elif λ == 0:
            λ = 1e-20
            # Smooth the counts by a tiny amount to avoid a problem where the M
            # step gets transition probabilities p(t | s) = 0/0 = nan for
            # context tags s that never occur at all, in particular s = EOS.
            # 
            # These 0/0 probabilities are never needed since those contexts
            # never occur.  So their value doesn't really matter ... except that
            # we do have to keep their value from being nan.  They show up in
            # the matrix version of the forward algorithm, where they are
            # multiplied by 0 and added into a sum.  A summand of 0 * nan would
            # regrettably turn the entire sum into nan.      

        self._save_time = time.time()      # mark start of training     
        dev_loss = loss(self)              # evaluate the model at the start of training
        old_dev_loss: float = dev_loss     # loss from the last epoch
        steps: int = 0   # total number of sentences the model has been trained on so far      
        while steps < max_steps:
            
            # E step: Run forward-backward on each sentence, and accumulate the
            # expected counts into self.A_counts, self.B_counts.
            #
            # Note: If you were using a GPU, you could get a speedup by running
            # forward-backward on several sentences in parallel.  This would
            # require writing the algorithm using higher-dimensional tensor
            # operations, allowing PyTorch to take advantage of hardware
            # parallelism.  For example, you'd update alpha[j-1] to alpha[j] for
            # all the sentences in the minibatch at once (with appropriate
            # handling for short sentences of length < j-1).  

            self._zero_counts()
            for sentence in tqdm(corpus, total=len(corpus), leave=True):
                isent = self._integerize_sentence(sentence, corpus)
                self.E_step(isent)
                steps += 1

            # M step: Update the parameters based on the accumulated counts.
            self.M_step(λ)
            if save_path: self.save(save_path, checkpoint=steps)  # save incompletely trained model in case we crash
            
            # Evaluate with the new parameters
            dev_loss = loss(self)   # this will print its own log messages
            if dev_loss >= old_dev_loss * (1-tolerance):
                # we haven't gotten much better, so perform early stopping
                break
            old_dev_loss = dev_loss            # remember for next eval batch
        
        # Save the trained model.
        if save_path: self.save(save_path)
  
    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> IntegerizedSentence:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            # Sentence comes from some other corpus that this HMM was not set up to handle.
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        return corpus.integerize_sentence(sentence)

    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet.
                
        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're integerizing
        correctly."""

        isent = self._integerize_sentence(sentence, corpus)
        if all(t is not None for _, t in isent):
            logp = torch.tensor(0.0, device=self.A.device)
            prev_t = self.bos_t
            for j in range(1, len(isent)):
                w_j, t_j = isent[j]
                logp = logp + torch.log(self.A[prev_t, t_j] + 1e-45)
                if w_j < self.V:
                    logp = logp + torch.log(self.B[t_j, w_j] + 1e-45)
                prev_t = t_j
            return logp
        # Un/partially tagged → use forward once you implement it
        return self.forward_pass(isent)

        # Integerize the words and tags of the given sentence, which came from the given corpus.
      

    


    def save(self, path: Path|str, checkpoint=None, checkpoint_interval: int = 300) -> None:
        """Save this model to the file named by path.  Or if checkpoint is not None, insert its 
        string representation into the filename and save to a temporary checkpoint file (but only 
        do this save if it's been at least checkpoint_interval seconds since the last save).  If 
        the save is successful, then remove the previous checkpoint file, if any."""

        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed

        now = time.time()
        old_save_time =           getattr(self, "_save_time", None)
        old_checkpoint_path =     getattr(self, "_checkpoint_path", None)
        old_total_training_time = getattr(self, "total_training_time", 0)

        if checkpoint is None:
            self._checkpoint_path = None   # this is a real save, not a checkpoint
        else:    
            if old_save_time is not None and now < old_save_time + checkpoint_interval: 
                return   # we already saved too recently to save another temp version
            path = path.with_name(f"{path.stem}-{checkpoint}{path.suffix}")  # use temp filename
            self._checkpoint_path = path

        # update the elapsed training time since we started training or last saved (if that happened)
        if old_save_time is not None:
            self.total_training_time = old_total_training_time + (now - old_save_time)
        del self._save_time
        
        # Save the model with the fields set as above, so that we'll 
        # continue from it correctly when we reload it.
        try:
            torch.save(self, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved model to {path}")
        except Exception as e:   
            # something went wrong with the save; so restore our old fields,
            # so that caller can potentially catch this exception and try again
            self._save_time          = old_save_time
            self._checkpoint_path    = old_checkpoint_path
            self.total_training_time = old_total_training_time
            raise e
        
        # Since save was successful, remember it and remove old temp version (if any)
        self._save_time = now
        if old_checkpoint_path: 
            try: os.remove(old_checkpoint_path)
            except FileNotFoundError: pass  # don't complain if the user already removed it manually

    @classmethod
    def load(cls, path: Path|str, device: str = 'cpu') -> HiddenMarkovModel:
        if isinstance(path, str): path = Path(path)   # convert str argument to Path if needed
            
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        model = torch.load(path, map_location=device)

        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls.__name__} but got {model.__class__.__name__} " \
                             f"from saved file {path}.")

        logger.info(f"Loaded model from {path}")
        return model