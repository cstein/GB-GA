def linear_threshold_modifier(value: float, threshold: float) -> float:
    """ Linearly increase threshold from 0 to 1 of a value

    :param value:
    :param threshold:
    :return:
    """
    return min(value, threshold) / threshold
