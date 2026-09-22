"""Linear-chain CRF over BIO tags (H4).

Motivation specific to SciEvent: the modal gap between two consecutive gold
argument spans is **0 tokens**, so the tagger must make `I-Context -> B-Method`
transitions with no O separator, and an independent per-token softmax has no
prior over those transitions. The dominant observed failure is *fragmentation* —
splitting one long gold span into two, which under IoU>0.5 one-to-one matching
can turn one true positive into zero matches plus two false positives. Learned
transition scores penalise exactly that pattern.
"""

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))  # [from, to]
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start, -0.1, 0.1)
        nn.init.uniform_(self.end, -0.1, 0.1)

    def _score(self, emissions, tags, mask):
        B, T, _ = emissions.shape
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for i in range(1, T):
            m = mask[:, i]
            s = self.transitions[tags[:, i - 1], tags[:, i]] + emissions[:, i].gather(1, tags[:, i: i + 1]).squeeze(1)
            score = score + s * m
        last = mask.sum(1).long() - 1
        score = score + self.end[tags.gather(1, last.unsqueeze(1)).squeeze(1)]
        return score

    def _logZ(self, emissions, mask):
        B, T, C = emissions.shape
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        for i in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions.unsqueeze(0), dim=1) + emissions[:, i]
            m = mask[:, i].unsqueeze(1)
            alpha = nxt * m + alpha * (1 - m)
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def forward(self, emissions, tags, mask):
        """Mean negative log-likelihood. `tags` must have no ignore values; use
        mask to zero out padding positions."""
        emissions = emissions.float()
        mask = mask.float()
        nll = self._logZ(emissions, mask) - self._score(emissions, tags, mask)
        return nll.sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi. Returns a list of tag lists, one per sequence."""
        emissions = emissions.float()
        B, T, C = emissions.shape
        mask = mask.bool()
        score = self.start.unsqueeze(0) + emissions[:, 0]
        history = []
        for i in range(1, T):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best, idx = broadcast.max(dim=1)
            nxt = best + emissions[:, i]
            m = mask[:, i].unsqueeze(1)
            score = torch.where(m, nxt, score)
            history.append(idx)
        score = score + self.end.unsqueeze(0)
        lengths = mask.sum(1).long()
        out = []
        for b in range(B):
            L = int(lengths[b])
            best = int(score[b].argmax())
            path = [best]
            for i in range(L - 2, -1, -1):
                best = int(history[i][b, best])
                path.append(best)
            out.append(path[::-1])
        return out

    @torch.no_grad()
    def marginals(self, emissions, mask):
        """Per-token posterior marginals via forward-backward (used for the
        confidence-based decoding rules and for posterior ensembling)."""
        emissions = emissions.float()
        B, T, C = emissions.shape
        m = mask.float()
        alpha = torch.empty(B, T, C, device=emissions.device)
        alpha[:, 0] = self.start.unsqueeze(0) + emissions[:, 0]
        for i in range(1, T):
            nxt = torch.logsumexp(alpha[:, i - 1].unsqueeze(2) + self.transitions.unsqueeze(0), 1) + emissions[:, i]
            mm = m[:, i].unsqueeze(1)
            alpha[:, i] = nxt * mm + alpha[:, i - 1] * (1 - mm)
        beta = torch.empty(B, T, C, device=emissions.device)
        beta[:, T - 1] = self.end.unsqueeze(0)
        for i in range(T - 2, -1, -1):
            nxt = torch.logsumexp(
                self.transitions.unsqueeze(0) + (emissions[:, i + 1] + beta[:, i + 1]).unsqueeze(1), 2
            )
            mm = m[:, i + 1].unsqueeze(1)
            beta[:, i] = nxt * mm + beta[:, i + 1] * (1 - mm)
        logp = alpha + beta
        return torch.softmax(logp, dim=-1)
