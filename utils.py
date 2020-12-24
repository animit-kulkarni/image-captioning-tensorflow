
def calc_max_length(tensor):
    """Find the maximum length of any caption in our dataset"""
    return max(len(t) for t in tensor)