from vaani.data.manifests import stable_hash


def assign(group_id: str) -> str:
    """80/10/10 by group. Group = speaker for speech, recording/video for noise,
    so a voice or a noise source can never straddle train and test."""
    b = stable_hash(group_id) % 10
    return "train" if b < 8 else ("val" if b == 8 else "test")
