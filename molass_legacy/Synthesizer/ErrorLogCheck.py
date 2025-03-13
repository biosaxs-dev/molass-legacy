"""
    ErrorLogCheck.py

    Copyright (c) 2024, Masatsuyo Takahashi, KEK-PF
"""

# borrowed from https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring
def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)

def all_known_errors(buffer):
    error_count = 0
    for i in findall("ERROR", buffer):
        error_count += 1
    
    known_error_count= 0
    for i in findall("ERROR,root,No counter info", buffer):
        known_error_count += 1
    
    return error_count == known_error_count