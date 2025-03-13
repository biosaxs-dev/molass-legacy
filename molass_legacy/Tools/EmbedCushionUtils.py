"""
    EmbedCushionUtils.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
def get_caller_attr(caller, attr_name, init_value):
    try:
        attr_object = caller.__getattribute__(attr_name)
    except:
        caller.__setattr__(attr_name, init_value)
        attr_object = init_value
    return attr_object
