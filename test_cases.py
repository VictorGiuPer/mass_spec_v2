# Format: (description, peaks, expected_num_boxes, expected_num_overlaps, expected_peaks_in_overlap or None)

TEST_CASES = [
    (
        "Two clearly separated peaks",
        [
            {"rt_center": 11.0, "mz_center": 151.0, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 20000},
            {"rt_center": 14.0, "mz_center": 158.0, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 25000}
        ],
        2, 0, None  # No overlap
    ),
    (
        "Strong overlap between two peaks",
        [
            {"rt_center": 13.0, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 18000},
            {"rt_center": 13.1, "mz_center": 154.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 45000}
        ],
        1, 1, 2  # Overlap of 2 peaks
    ),
    (
        "Four peaks: 2 overlap, 2 isolated",
        [
            {"rt_center": 11.0, "mz_center": 151.0, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 18000},
            {"rt_center": 11.1, "mz_center": 151.1, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 22000},
            {"rt_center": 13.5, "mz_center": 157.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 20000},
            {"rt_center": 14.0, "mz_center": 159.0, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 25000}
        ],
        3, 1, 2  # 2 overlapping peaks in 1 region
    ),
    (
        "Cluster of 3 overlapping peaks",
        [
            {"rt_center": 12.0, "mz_center": 153.0, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 20000},
            {"rt_center": 12.1, "mz_center": 153.1, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 22000},
            {"rt_center": 12.2, "mz_center": 153.2, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 21000}
        ],
        1, 1, 3  # One tight cluster of 3 overlapping peaks
    ),
    (
        "Close but not overlapping peaks",
        [
            {"rt_center": 13.0, "mz_center": 155.0, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 10000},
            {"rt_center": 13.4, "mz_center": 155.2, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 9500}
        ],
        2, 0, None  # Close but resolved as separate
    ),
    (
        "Intense + weak overlap",
        [
            {"rt_center": 12.5, "mz_center": 152.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 30000},
            {"rt_center": 12.6, "mz_center": 152.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 6000},
            {"rt_center": 14.5, "mz_center": 158.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 32000}
        ],
        2, 1, 2  # First two peaks overlap; third is isolated
    ),
    (
        "Five peaks: 3 spaced, 2 overlapping",
        [
            {"rt_center": 10.5, "mz_center": 150.5, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10000},
            {"rt_center": 11.0, "mz_center": 151.0, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10000},
            {"rt_center": 11.5, "mz_center": 151.5, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10000},
            {"rt_center": 13.0, "mz_center": 154.0, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 22000},
            {"rt_center": 13.05, "mz_center": 154.05, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 21000}
        ],
        4, 1, 2  # Last two overlap, others are spaced
    )
]
