__all__ = ["ActorCritic", "PPOLogic", "RolloutBuffer"]


def __getattr__(name):
    """Load PyTorch components only when they are explicitly requested."""
    if name == "ActorCritic":
        from .network import ActorCritic

        return ActorCritic
    if name == "PPOLogic":
        from .logic import PPOLogic

        return PPOLogic
    if name == "RolloutBuffer":
        from .buffer import RolloutBuffer

        return RolloutBuffer
    raise AttributeError(name)
