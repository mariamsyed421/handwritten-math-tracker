# cleanup.py: Fixes TrOCR's common symbol mistakes we have seen so far
# + checks whether the cleaned text looks like a valid equation step.

FIXES = {  # misread char: what it should be, based on common mistakes we saw
    "t": "+", "T": "+",
    "X": "x",
    "o": "0", "O": "0",
    "l": "1", "I": "1", "|": "1",
    "S": "5", "s": "5",
}
ALLOWED = set("0123456789x+-=()/*^")  # every character a math step is allowed to have

def clean(text):
    s = text.replace(" ", "")  # TrOCR adds spaces between characters, remove them
    s = "".join(FIXES.get(c, c) for c in s)  # swap known misreads, keep everything else
    if "=" not in s:
        # we observed trOCR to largely misread "="
        # find the last character that is probably a misread =
        # loop goes backwards, from the end of the string to the start
        for i in range(len(s) - 1, -1, -1):
            if s[i] in "-._":
                s = s[:i] + "=" + s[i + 1:]  # rebuild the string with = in that spot
                break  # only replace one time
    return s

def looks_valid(s): #check if it seems like a valid equation step
    if s.count("=") != 1:
        return False  # a step should have one = only
    left, right = s.split("=")  # the two sides of the equation
    if left == "" or right == "":
        return False  # nothing on one side of the =
    return all(c in ALLOWED for c in s)  # no leftover non-allowed characters


if __name__ == "__main__":
    # test cases: 
    for raw in ["2 x t 4 - 1 0", "2x. 6", "X _ 3", "x - 2 . 5", "2 x = 1 O", "2x6", "x ? 3"]:
        s = clean(raw)
        print(repr(raw), "->", s, "| valid:", looks_valid(s)) 