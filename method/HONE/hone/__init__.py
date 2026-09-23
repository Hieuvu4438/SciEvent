"""HONE — Hard-negative Out-of-fold Neural vErification.

Propose-then-verify extraction for SciEvent. A clause-level span tagger (the
CARVE tagger, vendored here as the *proposer*) over-generates candidate argument
spans from several seeds; a learned cross-encoder *verifier*, trained on
out-of-fold proposals so it sees the proposer's real mistakes, decides which
candidates to keep and which role each carries.
"""
__version__ = "0.1.0"
