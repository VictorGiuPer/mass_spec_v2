# Format: (description, peaks, expected_num_boxes, expected_num_overlaps, expected_peaks_in_overlap or None)

TEST_CASES = [
    (
        "Variant - Two clearly separated peaks 1",
        [
            {"rt_center": 10.8, "mz_center": 150.8, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 21000},
            {"rt_center": 14.2, "mz_center": 158.2, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 24000}
        ],
        2, 0, None
    ),
    (
        "Variant - Two clearly separated peaks 2",
        [
            {"rt_center": 11.1, "mz_center": 151.2, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 19000},
            {"rt_center": 13.9, "mz_center": 157.9, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 25500}
        ],
        2, 0, None
    ),
    (
        "Variant - Two clearly separated peaks 3",
        [
            {"rt_center": 10.9, "mz_center": 150.9, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 20500},
            {"rt_center": 14.1, "mz_center": 158.1, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 25000}
        ],
        2, 0, None
    ),
    (
        "Variant - Two clearly separated peaks 4",
        [
            {"rt_center": 11.3, "mz_center": 151.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 20000},
            {"rt_center": 13.7, "mz_center": 157.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 24500}
        ],
        2, 0, None
    ),
    (
        "Variant - Two clearly separated peaks 5",
        [
            {"rt_center": 10.7, "mz_center": 150.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 19500},
            {"rt_center": 14.3, "mz_center": 158.3, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 24800}
        ],
        2, 0, None
    ),

###########################################################################################################

    (
        "Variant - Strong overlap between two peaks 1",
        [
            {"rt_center": 13.0, "mz_center": 154.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 17000},
            {"rt_center": 13.05, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 46000}
        ],
        1, 1, 2
    ),
    (
        "Variant - Strong overlap between two peaks 2",
        [
            {"rt_center": 13.2, "mz_center": 154.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 19000},
            {"rt_center": 13.3, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 44000}
        ],
        1, 1, 2
    ),
    (
        "Variant - Strong overlap between two peaks 3",
        [
            {"rt_center": 12.9, "mz_center": 154.3, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 18000},
            {"rt_center": 13.0, "mz_center": 154.2, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 46000}
        ],
        1, 1, 2
    ),
    (
        "Variant - Strong overlap between two peaks 4",
        [
            {"rt_center": 13.1, "mz_center": 154.7, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 20000},
            {"rt_center": 13.2, "mz_center": 154.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 43000}
        ],
        1, 1, 2
    ),
    (
        "Variant - Strong overlap between two peaks 5",
        [
            {"rt_center": 13.0, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 18500},
            {"rt_center": 13.15, "mz_center": 154.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 47000}
        ],
        1, 1, 2
    ),

###########################################################################################################

    (
        "Variant - Four peaks: 2 overlap, 2 isolated 1",
        [
            {"rt_center": 11.0, "mz_center": 151.0, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 17500},
            {"rt_center": 11.1, "mz_center": 151.1, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 22500},
            {"rt_center": 13.6, "mz_center": 157.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 20500},
            {"rt_center": 14.1, "mz_center": 159.1, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 24000}
        ],
        3, 1, 2
    ),
    (
        "Variant - Four peaks: 2 overlap, 2 isolated 2",
        [
            {"rt_center": 11.2, "mz_center": 151.2, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 18000},
            {"rt_center": 11.3, "mz_center": 151.3, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 21000},
            {"rt_center": 13.4, "mz_center": 157.3, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 19000},
            {"rt_center": 14.0, "mz_center": 158.9, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 25500}
        ],
        3, 1, 2
    ),
    (
        "Variant - Four peaks: 2 overlap, 2 isolated 3",
        [
            {"rt_center": 10.9, "mz_center": 150.9, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 18500},
            {"rt_center": 11.0, "mz_center": 151.0, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 21500},
            {"rt_center": 13.5, "mz_center": 157.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 19500},
            {"rt_center": 14.2, "mz_center": 159.2, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 25000}
        ],
        3, 1, 2
    ),
    (
        "Variant - Four peaks: 2 overlap, 2 isolated 4",
        [
            {"rt_center": 11.1, "mz_center": 151.1, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 19000},
            {"rt_center": 11.2, "mz_center": 151.2, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 23000},
            {"rt_center": 13.6, "mz_center": 157.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 19800},
            {"rt_center": 14.3, "mz_center": 159.3, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 24500}
        ],
        3, 1, 2
    ),
    (
        "Variant - Four peaks: 2 overlap, 2 isolated 5",
        [
            {"rt_center": 11.0, "mz_center": 150.9, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 17000},
            {"rt_center": 11.15, "mz_center": 151.0, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 20000},
            {"rt_center": 13.7, "mz_center": 157.7, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 21000},
            {"rt_center": 14.4, "mz_center": 159.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 26000}
        ],
        3, 1, 2
    ),

###########################################################################################################

    (
        "Variant - Cluster of 3 overlapping peaks 1",
        [
            {"rt_center": 12.0, "mz_center": 153.0, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 20500},
            {"rt_center": 12.1, "mz_center": 153.1, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 22500},
            {"rt_center": 12.2, "mz_center": 153.2, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 21500}
        ],
        1, 1, 3
    ),
    (
        "Variant - Cluster of 3 overlapping peaks 2",
        [
            {"rt_center": 11.9, "mz_center": 152.9, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 21000},
            {"rt_center": 12.0, "mz_center": 153.0, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 23000},
            {"rt_center": 12.1, "mz_center": 153.1, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 22000}
        ],
        1, 1, 3
    ),
    (
        "Variant - Cluster of 3 overlapping peaks 3",
        [
            {"rt_center": 12.1, "mz_center": 153.1, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 20000},
            {"rt_center": 12.2, "mz_center": 153.2, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 21500},
            {"rt_center": 12.3, "mz_center": 153.3, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 22500}
        ],
        1, 1, 3
    ),
    (
        "Variant - Cluster of 3 overlapping peaks 4",
        [
            {"rt_center": 11.95, "mz_center": 152.95, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 19500},
            {"rt_center": 12.05, "mz_center": 153.05, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 21000},
            {"rt_center": 12.15, "mz_center": 153.15, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 20500}
        ],
        1, 1, 3
    ),
    (
        "Variant - Cluster of 3 overlapping peaks 5",
        [
            {"rt_center": 12.05, "mz_center": 153.05, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 21500},
            {"rt_center": 12.15, "mz_center": 153.15, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 22000},
            {"rt_center": 12.25, "mz_center": 153.25, "rt_sigma": 0.3, "mz_sigma": 0.03, "amplitude": 20000}
        ],
        1, 1, 3
    ),

###########################################################################################################
    
    (
        "Variant - Close but not overlapping peaks 1",
        [
            {"rt_center": 13.0, "mz_center": 155.0, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 10500},
            {"rt_center": 13.5, "mz_center": 155.3, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 9800}
        ],
        2, 0, None
    ),
    (
        "Variant - Close but not overlapping peaks 2",
        [
            {"rt_center": 13.1, "mz_center": 155.1, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 10200},
            {"rt_center": 13.6, "mz_center": 155.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 9600}
        ],
        2, 0, None
    ),
    (
        "Variant - Close but not overlapping peaks 3",
        [
            {"rt_center": 12.9, "mz_center": 154.9, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 10000},
            {"rt_center": 13.4, "mz_center": 155.2, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 9400}
        ],
        2, 0, None
    ),
    (
        "Variant - Close but not overlapping peaks 4",
        [
            {"rt_center": 13.2, "mz_center": 155.2, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 11000},
            {"rt_center": 13.7, "mz_center": 155.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 9200}
        ],
        2, 0, None
    ),
    (
        "Variant - Close but not overlapping peaks 5",
        [
            {"rt_center": 13.0, "mz_center": 155.1, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 9700},
            {"rt_center": 13.6, "mz_center": 155.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 9900}
        ],
        2, 0, None
    ),

###########################################################################################################

    (
        "Variant - Intense + weak overlap 1",
        [
            {"rt_center": 12.4, "mz_center": 152.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 31000},
            {"rt_center": 12.5, "mz_center": 152.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 7000},
            {"rt_center": 14.6, "mz_center": 158.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 31500}
        ],
        2, 1, 2
    ),
    (
        "Variant - Intense + weak overlap 2",
        [
            {"rt_center": 12.6, "mz_center": 152.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 29500},
            {"rt_center": 12.7, "mz_center": 152.7, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 6500},
            {"rt_center": 14.4, "mz_center": 158.4, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 33000}
        ],
        2, 1, 2
    ),
    (
        "Variant - Intense + weak overlap 3",
        [
            {"rt_center": 12.5, "mz_center": 152.6, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 32000},
            {"rt_center": 12.6, "mz_center": 152.7, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 5500},
            {"rt_center": 14.7, "mz_center": 158.7, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 31000}
        ],
        2, 1, 2
    ),
    (
        "Variant - Intense + weak overlap 4",
        [
            {"rt_center": 12.45, "mz_center": 152.45, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 30500},
            {"rt_center": 12.55, "mz_center": 152.55, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 6000},
            {"rt_center": 14.55, "mz_center": 158.55, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 32500}
        ],
        2, 1, 2
    ),
    (
        "Variant - Intense + weak overlap 5",
        [
            {"rt_center": 12.55, "mz_center": 152.55, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 29800},
            {"rt_center": 12.65, "mz_center": 152.65, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 5800},
            {"rt_center": 14.45, "mz_center": 158.45, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 31800}
        ],
        2, 1, 2
    ),

###########################################################################################################

    (
        "Variant - Five peaks: 3 spaced, 2 overlapping 1",
        [
            {"rt_center": 10.6, "mz_center": 150.6, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 9500},
            {"rt_center": 11.1, "mz_center": 151.1, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10200},
            {"rt_center": 11.6, "mz_center": 151.6, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10800},
            {"rt_center": 13.1, "mz_center": 154.1, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 23000},
            {"rt_center": 13.2, "mz_center": 154.2, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 22000}
        ],
        4, 1, 2
    ),
    (
        "Variant - Five peaks: 3 spaced, 2 overlapping 2",
        [
            {"rt_center": 10.4, "mz_center": 150.4, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 9800},
            {"rt_center": 10.9, "mz_center": 150.9, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10300},
            {"rt_center": 11.4, "mz_center": 151.4, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10700},
            {"rt_center": 13.3, "mz_center": 154.3, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 22500},
            {"rt_center": 13.4, "mz_center": 154.4, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 21500}
        ],
        4, 1, 2
    ),
    (
        "Variant - Five peaks: 3 spaced, 2 overlapping 3",
        [
            {"rt_center": 10.7, "mz_center": 150.7, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10100},
            {"rt_center": 11.2, "mz_center": 151.2, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 9900},
            {"rt_center": 11.7, "mz_center": 151.7, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10500},
            {"rt_center": 13.2, "mz_center": 154.2, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 21800},
            {"rt_center": 13.3, "mz_center": 154.3, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 20500}
        ],
        4, 1, 2
    ),
    (
        "Variant - Five peaks: 3 spaced, 2 overlapping 4",
        [
            {"rt_center": 10.5, "mz_center": 150.6, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 9500},
            {"rt_center": 11.0, "mz_center": 151.1, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 9700},
            {"rt_center": 11.5, "mz_center": 151.6, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 9800},
            {"rt_center": 13.0, "mz_center": 154.0, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 23500},
            {"rt_center": 13.1, "mz_center": 154.1, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 22500}
        ],
        4, 1, 2
    ),
    (
        "Variant - Five peaks: 3 spaced, 2 overlapping 5",
        [
            {"rt_center": 10.8, "mz_center": 150.8, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10000},
            {"rt_center": 11.3, "mz_center": 151.3, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 9800},
            {"rt_center": 11.8, "mz_center": 151.8, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 10200},
            {"rt_center": 13.5, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 21500},
            {"rt_center": 13.6, "mz_center": 154.6, "rt_sigma": 0.2, "mz_sigma": 0.03, "amplitude": 21000}
        ],
        4, 1, 2
    ),

]
