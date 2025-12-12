# Temporary file!
# used to store this function in the mean Time

def increment(counter, limits):
    """
    Returns the updated counter list,
    or False if we reached the final combination.
    """
    i = 0
    counter[i] += 1

    while i < len(counter) and counter[i] >= limits[i]:
        counter[i] = 0
        i += 1
        if i < len(counter):
            counter[i] += 1

    # If we overflowed past the last digit
    if i == len(counter):
        return False

    return counter

if __name__ == "__main__":
    len_counter = 0
    counter = [0] * len_counter       # 3-digit counter
    limits  = [96] * len_counter      # digit 0 goes 0 to 9, digit 1 goes 0 to 4, digit 2 goes 0 to 2