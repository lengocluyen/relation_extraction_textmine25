from typing import Any, Dict


def replace_e1_mentions_by_e2_ones(
    text: str, e1: Dict[str, Any], e2: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """generate all the possible entities obtained by replacing all the mentions
    of entity e1 by the mentions of e2.

    Args:
        text (str): the original text with mentions of e1.
        e1 (Dict[str, Any]): an entity
        e2 (Dict[str, Any]): another entity.

    Returns:
        Dict[str, Dict[str, Any]]: all generated texts and their new entities
            {generated_text: entities }
    """
    pass
