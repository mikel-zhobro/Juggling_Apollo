import itertools

def to_bitstring(tup, length):
    # in this case, length is equivalent to the maxmimum throw height
    result = [0] * length
    for pos in tup:
        result[length - pos - 1] = 1
    return tuple(result)

def get_next_state(current, throw):
    # current here is an integer instead of an array, so we are bitshifting the integer itself. We could rotate a tuple, but rightshift looks much cleaner in code than some kind of special array rotate function.
    # A 0 throw is a bit of a special case, since instead of adding a new ball, you don’t add anything new.
    # This doesn’t check for anything being invalid
    if throw > 0:
        return (current >> 1) | 2**(throw - 1)
    else:
        return current >> 1


def get_valid_tosses(state, height):
    # find all valid future states from our current state
    state = list(state)
    balls = sum(state)
    if len(state) > height:
        raise ValueError("You can't throw that high")
    elif len(state) < height:
        state = [0]*(height - len(state)) + state  # extend the length so that
                                                   # length == height
    state = [0] + state[:-1]
    if sum(state) == balls:
        return [0]

    valid_tosses = [len(state) - 1 for i, pos in enumerate(state) if not pos]
    return valid_tosses



max_height = 3
balls = 3
print(list(itertools.combinations(range(max_height), balls)))
print(to_bitstring((0,1,2), max_height))


print(get_valid_tosses((0,0,1,1,1), 5))
get_valid_tosses((0,0,1,1,1), 5) == [3,4,5]
get_valid_tosses((1,0,1,0,1), 5) == [1,3,5]
get_valid_tosses((1,1,1,0,0), 5) == [0]