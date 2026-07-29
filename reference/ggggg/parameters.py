mev_to_ev = 1e-3

onsite_and_strand_hopping = {
    "G": {"e_h": -4278.0, "e_l": 1137.0, "t_l": 19.0, "t_h": -115.0},
    "C": {"e_h": -6519.0, "e_l": -1065.0, "t_l": -61.0, "t_h": -24.0},
    "A": {"e_h": -5245.0, "e_l": 259.0, "t_l": 24.0, "t_h": 21.0},
    "T": {"e_h": -6298.0, "e_l": -931.0, "t_l": -23.0, "t_h": -98.0},
}

base_pairs = {
    "gc": ("G", "C"),
    "at": ("A", "T"),
}

nearest_cross_hopping = {
    "gc": {"homo": 2.0, "lumo": 63.0},
    "at": {"homo": 26.0, "lumo": 34.0},
}

def to_ev(value_mev):
    return value_mev * mev_to_ev

def validate_band(band):
    if band not in ("homo", "lumo"):
        raise ValueError("band must be homo or lumo")

def validate_pair(pair):
    if pair not in base_pairs:
        raise ValueError("pair must be gc or at")

def get_base_pair(pair):
    pair = pair.lower()
    validate_pair(pair)
    return base_pairs[pair]

def get_onsite(base, band):
    validate_band(band)
    if band == "homo":
        key = "e_h"
    else:
        key = "e_l"
    return to_ev(onsite_and_strand_hopping[base][key])

def get_strand_hopping(base, band):
    validate_band(band)
    if band == "homo":
        key = "t_h"
    else:
        key = "t_l"
    return to_ev(onsite_and_strand_hopping[base][key])

def get_nearest_cross_hopping(pair, band):
    pair = pair.lower()
    validate_pair(pair)
    validate_band(band)
    return to_ev(nearest_cross_hopping[pair][band])
