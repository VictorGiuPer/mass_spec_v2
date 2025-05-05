def delta_bic_to_confidence_score(delta_bic, saturation=20):
    return min(1.0, delta_bic / saturation) * 100
