"""AutoFit module for Block 3 model selection."""

# Core imports (safe)
from .diagnose_dataset import diagnose_dataset
from .rule_based_composer import RuleBasedComposer, ComposerConfig, compose_from_profile
from .two_stage_selector import TwoStageSelector, select_from_profile

# Optional imports (may have syntax issues in legacy code)
try:
    from .compose_candidates import compose_candidates
except (SyntaxError, ImportError):
    compose_candidates = None

try:
    from .budget_search import successive_halving
except (SyntaxError, ImportError):
    successive_halving = None

try:
    from .leaderboard import write_leaderboard
except (SyntaxError, ImportError):
    write_leaderboard = None

__all__ = [
    "diagnose_dataset",
    "RuleBasedComposer",
    "ComposerConfig",
    "compose_from_profile",
    "TwoStageSelector",
    "select_from_profile",
    "compose_candidates",
    "successive_halving",
    "write_leaderboard",
]
