def flatten(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            for bar in flatten(k):
                yield bar
