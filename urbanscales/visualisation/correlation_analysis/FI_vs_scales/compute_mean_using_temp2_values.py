feature_ranges = {
    "betweenness": {"mean": 0.04597491691686059, "max": 0.05108693180930041},
    "metered_count": {
        "mean": [0.31259602378059487, 0.2849257282843944, 0.32897317059036524, 0.12866392828905227], 
        "max": [0.6293410191788906, 0.07684809604088638, 0.07546141720711702, 0.29031890533200455]
    },
    "total_crossings": {
        "mean": [0.1583207893919382, 0.16450482862432897, 0.14154647230925949], 
        "max": [0.4826918917401068, 0.20801395798512212, 0.40450592852338646]
    },
    "street_length_total": {
        "mean": [0.08712109402733148, 0.07021212422246578], 
        "max": [0.14275479244030853, 0.05793771121283378]
    },
    "global_betweenness": {"mean": 0.028538748836211533, "max": 0.03004274647575285},
    "k_avg": {"mean": 0.045280616993588776, "max": 0.04488598093563059},
    "circuity_avg": {"mean": 0.03650493456168592, "max": 0.06307857318043444},
}

# Example usage:
# To get the mean range of metered_count
mean_metered_count = feature_ranges["metered_count"]["mean"]

# To get the max range of total_crossings
max_total_crossings = feature_ranges["total_crossings"]["max"]

