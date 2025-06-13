import numpy as np
from sklearn.cluster import DBSCAN

class LCMS_RidgeWalker:


    # PUBLIC INTERFACE
    def __init__(self, min_intensity=6e4, min_slope=0.5, fusion_threshold=0.7, weights=None):
        """
        Initialize RidgeWalker with thresholds and scoring weights.

        Args:
            min_intensity (float): Minimum signal level to consider in ridge tracking.
            min_slope (float): Minimum d/dt slope required to extend a ridge.
            fusion_threshold (float): Threshold to confirm overlap between ridges.
            weights (dict): Weighting for overlap scoring components.
        """
        self.min_intensity = min_intensity
        self.min_slope = min_slope
        self.fusion_threshold = fusion_threshold
        self.weights = weights if weights else {"width": 0.2, "smoothness": 0.3, "fork": 0.5}

    def fit(self, grid, d_rt, dd_rt, mz_axis=None, rt_axis=None):
        """
        Run ridge detection and overlap analysis on a given grid.

        Args:
            grid (2D array): Intensity grid.
            d_rt (2D array): First derivative w.r.t. RT.
            dd_rt (2D array): Second derivative w.r.t. RT.
            mz_axis (1D array): m/z values for rows.
            rt_axis (1D array): RT values for columns.

        Returns:
            dict: Result summary containing peaks and overlap info.
        """
        self.grid = grid
        self.d_rt = d_rt
        self.dd_rt = dd_rt
        self.mz_axis = np.array(mz_axis)
        self.rt_axis = np.array(rt_axis)
        self.ridges = self._track_all_ridges()
        self.overlap_summary = self._analyze_ridge_pairs()
        return self._build_result()
    
    def get_peak_maxima_from_ridges(self, ridge_indices, n_peaks=1):
        """
        Return top N peak maxima from selected ridges.

        Args:
            ridge_indices (list[int]): Indices of ridges to analyze.
            n_peaks (int): Number of peak maxima to return.

        Returns:
            list: Top N (mz, rt) peak coordinates sorted by intensity.
        """
        peaks = []

        for i in ridge_indices:
            ridge = self.ridges[i]
            if not ridge:
                continue

            max_point = max(ridge, key=lambda pt: self.grid[pt[0], pt[1]])
            mz_idx, rt_idx = max_point
            intensity = self.grid[mz_idx, rt_idx]
            mz = self.mz_axis[mz_idx]
            rt = self.rt_axis[rt_idx]
            peaks.append(((mz, rt), intensity))

        # Sort by intensity and keep top N
        peaks.sort(key=lambda x: -x[1])
        return [pt for pt, _ in peaks[:n_peaks]]

    def _build_result(self):
        """
        Assemble final result dictionary using overlap analysis and ridge maxima.

        Returns:
            dict: Summary with peaks, overlaps, and ridge metadata.
        """
        summary = self.overlap_summary
        peak_locations = []

        # Use ridges from overlaps if any
        peak_ridge_indices = set()
        if summary and summary.get("overlap_details"):
            for overlap in summary["overlap_details"]:
                peak_ridge_indices.update(overlap["ridge_pair"])

        # If no overlaps or none detected, fall back to top ridges
        if not peak_ridge_indices:
            for i, ridge in enumerate(self.ridges):
                if len(ridge) >= 3:
                    intensities = [self.grid[mz, rt] for mz, rt in ridge]
                    if max(intensities) >= self.min_intensity:
                        peak_ridge_indices.add(i)

        # Extract max point per ridge
        for i in peak_ridge_indices:
            ridge = self.ridges[i]
            if not ridge:
                continue
            max_mz, max_rt = max(ridge, key=lambda pt: self.grid[pt[0], pt[1]])
            peak_locations.append((float(self.mz_axis[max_mz]), float(self.rt_axis[max_rt])))

        peak_locations = list(set(peak_locations))

        return {
            "overlap_detected": summary.get("overlap_detected", False) if summary else False,
            "num_overlap_events": summary.get("num_overlap_events", 0) if summary else 0,
            "num_peaks_in_overlap": len(peak_locations),
            "num_ridges_tracked": len(self.ridges),
            "ridge_pairs_analyzed": summary.get("num_overlap_events", 0) if summary else 0,
            "overlap_details": summary.get("overlap_details", []) if summary else [],
            "peak_locations": peak_locations
        }


    # RIDGE DETECTION CORE
    def _find_local_maxima(self, column):
        """
        Find local maxima in a 1D column (e.g., m/z slice at fixed RT).

        Args:
            column (1D array): Intensity values.

        Returns:
            list[int]: Indices of local maxima above min_intensity.
        """
        maxima = set()

        # First pass: top to bottom
        for i in range(1, len(column) - 1):
            if column[i] > self.min_intensity and column[i] > column[i - 1] and column[i] > column[i + 1]:
                maxima.add(i)

        # Second pass: bottom to top — gives weaker peaks a chance
        for i in range(len(column) - 2, 0, -1):
            if column[i] > self.min_intensity and column[i] > column[i - 1] and column[i] > column[i + 1]:
                maxima.add(i)  # set avoids duplicates

        return list(maxima)

    def _track_single_ridge(self, mz, rt):
        """
        Track a single ridge from a local maximum in both directions.

        Args:
            mz (int): Starting m/z index.
            rt (int): Starting RT index.

        Returns:
            list[(int, int)]: List of (mz, rt) points along the ridge.
        """
        backward = self._extend_ridge_one_direction(mz, rt, -1)[::-1]
        forward = self._extend_ridge_one_direction(mz, rt, 1)
        ridge = backward + [(mz, rt)] + forward

        # Early contrast check to skip flat ridges
        if len(ridge) >= 3:
            intensities = [self.grid[mz, rt] for mz, rt in ridge]
            contrast = (max(intensities) - min(intensities)) / (max(intensities) + 1e-9)
            if contrast < 0.05:
                return []

        return ridge

    def _extend_ridge_one_direction(self, mz, rt, direction):
        """
        Extend a ridge path in the specified RT direction.

        Args:
            mz (int): Starting m/z.
            rt (int): Starting RT.
            direction (int): +1 for forward, -1 for backward.

        Returns:
            list[(int, int)]: Path of ridge in the given direction.
        """
        path = []
        mz_len, rt_len = self.grid.shape
        current_mz, current_rt = mz, rt

        while 0 <= current_rt + direction < rt_len:
            next_rt = current_rt + direction
            mz_window = range(max(0, current_mz - 1), min(mz_len, current_mz + 2))

            # Track the top N intensities (we can change N if needed)
            candidates = []
            for m in mz_window:
                intensity = self.grid[m, next_rt]
                slope_idx = next_rt if direction == -1 else next_rt - 1
                if 0 <= slope_idx < self.d_rt.shape[1]:
                    slope = self.d_rt[m, slope_idx]
                    if ((direction == 1 and slope >= self.min_slope) or
                        (direction == -1 and slope <= -self.min_slope)) and intensity >= self.min_intensity:
                        candidates.append((m, intensity))
            
            if not candidates:
                break

            # Sort candidates by intensity and extend multiple paths
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:1]  # top 2 candidates

            # Add new paths for both top candidates
            for candidate_mz, _ in candidates:
                path.append((candidate_mz, next_rt))
                current_mz = candidate_mz
                current_rt = next_rt

        return path

    def _track_all_ridges(self):
        """
        Track all local maxima across RT and build candidate ridges.

        Returns:
            list[list[(int, int)]]: All accepted ridge paths.
        """
        ridge_paths = []
        visited = np.zeros_like(self.grid, dtype=bool)
        mz_len, rt_len = self.grid.shape
        for rt in range(rt_len):
            local_maxima = self._find_local_maxima(self.grid[:, rt])
            # Sort local maxima by intensity to prioritize stronger peaks
            local_maxima = sorted(local_maxima, key=lambda m: self.grid[m, rt], reverse=True)

            for mz in local_maxima:
                if visited[mz, rt]:
                    continue
                # Track multiple possible ridges instead of just one
                paths_to_track = [self._track_single_ridge(mz, rt)]
                while paths_to_track:  # As long as there are paths to track
                    path = paths_to_track.pop(0)  # Get the next path to extend

                    # Filter short or jumpy ridges
                    if len(path) < 3:
                        continue
                    mz_jump = [abs(path[i + 1][0] - path[i][0]) for i in range(len(path) - 1)]
                    if np.percentile(mz_jump, 90) > 0.2:
                        continue


                    if len(path) > 1:
                        ridge_paths.append(path)
                        # Add visited coordinates to avoid re-tracking
                        for m, r in path:
                            for dm in range(-1, 2):
                                for dr in range(-1, 2):
                                    mm, rr = m + dm, r + dr
                                    if 0 <= mm < self.grid.shape[0] and 0 <= rr < self.grid.shape[1]:
                                        visited[mm, rr] = True

        return ridge_paths

    # RIDGE FEATURE EXTRACTION
    def _ridge_width(self, ridge):
        """
        Estimate the width of a ridge at half max intensity.

        Args:
            ridge (list): Sequence of (mz, rt) points.

        Returns:
            float: Median width of the ridge in m/z index units.
        """
        widths = []
        for mz, rt in ridge:
            intensity = self.grid[mz, rt]
            half_max = intensity / 2
            l, r = mz, mz
            while l > 0 and self.grid[l, rt] > half_max:
                l -= 1
            while r < self.grid.shape[0] - 1 and self.grid[r, rt] > half_max:
                r += 1
            widths.append(r - l)
        return np.median(widths)

    def _intensity_smoothness(self, ridge):
        """
        Compute intensity smoothness based on std dev of differences.

        Args:
            ridge (list): Ridge path.

        Returns:
            float: Standard deviation of first differences in intensity.
        """
        intensities = [self.grid[mz, rt] for mz, rt in ridge]
        return np.std(np.diff(intensities))

    def _fork_sharpness(self, ridge1, ridge2):
        """
        Evaluate second derivative difference between overlapping ridge sections.

        Args:
            ridge1, ridge2 (list): Two ridge paths.

        Returns:
            float: Sharpness score based on dd_rt differences and directional contrast.
        """
        overlap = set(rt for _, rt in ridge1) & set(rt for _, rt in ridge2)
        sharp_diffs = []
        consistent_signs = 0
        for rt in overlap:
            mz1 = next(mz for mz, r in ridge1 if r == rt)
            mz2 = next(mz for mz, r in ridge2 if r == rt)
            dd1 = self.dd_rt[mz1, rt] if 0 <= mz1 < self.dd_rt.shape[0] else 0
            dd2 = self.dd_rt[mz2, rt] if 0 <= mz2 < self.dd_rt.shape[0] else 0
            sharp_diffs.append(abs(dd1 - dd2))
            if np.sign(dd1) != np.sign(dd2):
                consistent_signs += 1
        if not sharp_diffs:
            return 0
        sharp_score = np.mean(sharp_diffs)
        direction_bonus = consistent_signs / len(overlap) if overlap else 0
        return sharp_score * (1 + direction_bonus)

    def _valley_score_between_ridges(self, ridge1, ridge2):
        """
        Evaluate valley depth and asymmetry between two ridges.

        Args:
            ridge1, ridge2 (list): Two ridge paths.

        Returns:
            dict | None: Valley score metrics, or None if comparison is invalid.
        """
        r1, r2 = {rt: mz for mz, rt in ridge1}, {rt: mz for mz, rt in ridge2}
        overlap_rts = sorted(set(r1) & set(r2))
        if len(overlap_rts) < 3:
            return None
        valleys, r1_ints, r2_ints = [], [], []
        for rt in overlap_rts:
            mz1, mz2 = r1[rt], r2[rt]
            if mz1 == mz2: continue
            mmin, mmax = sorted([mz1, mz2])
            valley_line = self.grid[mmin+1:mmax, rt] if mmax - mmin > 1 else []
            if len(valley_line) > 0:
                valleys.append(np.min(valley_line))
            r1_ints.append(self.grid[mz1, rt])
            r2_ints.append(self.grid[mz2, rt])
        if not valleys:
            return None
        valley = np.median(valleys)
        max1, max2 = np.max(r1_ints), np.max(r2_ints)
        score = min(max1, max2) - valley
        norm_score = score / max(max1, max2)
        asymmetry = abs(max1 - max2) / (max1 + max2 + 1e-9)
        confidence = norm_score * np.exp(-asymmetry * 2)
        return {
            "overlap_rt": overlap_rts,
            "ridge1_max": max1,
            "ridge2_max": max2,
            "valley_intensity": valley,
            "score": score,
            "norm_score": norm_score,
            "asymmetry": asymmetry,
            "confidence": confidence
        }


    # OVERLAP ANALYSIS
    def _fusion_score(self, width_diff, smooth_diff, fork_sharp):
        """
        Compute a composite score for overlap likelihood between two ridges.

        Args:
            width_diff (float): Absolute width difference.
            smooth_diff (float): Absolute smoothness difference.
            fork_sharp (float): Fork sharpness score.

        Returns:
            float: Weighted fusion score.
        """
        w = self.weights
        norm_width = 1 / (1 + width_diff)
        norm_smooth = 1 / (1 + smooth_diff)
        norm_fork = np.tanh(fork_sharp)  # still fine as-is
        return (w["width"] * norm_width +
                w["smoothness"] * norm_smooth +
                w["fork"] * norm_fork)

    def _analyze_ridge_pairs(self):
        """
        Analyze all pairs of ridges for potential overlap.

        Returns:
            dict: Summary of overlaps and deduplicated peak estimates.
        """
        labels = []
        ridge_scores = []
        for i in range(len(self.ridges)):
            ridge = self.ridges[i]
            if len(ridge) < 3:
                continue  # too short
            intensities = [self.grid[mz, rt] for mz, rt in ridge]
            
            if max(intensities) < self.min_intensity:
                print(f"[RIDGE FILTERED] Ridge {i} too weak: max_intensity={max(intensities):.1f}")
                continue

            contrast_ratio = (max(intensities) - min(intensities)) / (max(intensities) + 1e-9)
            if contrast_ratio < 0.03:
                print(f"[RIDGE FILTERED] Ridge {i} low contrast: ratio={contrast_ratio:.2f}")
                continue  # low contrast

            rt_spread = max(rt for _, rt in ridge) - min(rt for _, rt in ridge)
            if rt_spread < 2:
                print(f"[RIDGE FILTERED] Ridge {i} short RT span: ΔRT={rt_spread:.1f}")
                continue  # too short in time (optional but helps)

            ridge_scores.append((i, ridge))

        def is_mid_saddle(ridge_i, other_ridges):
            """Check if a ridge lies between two stronger ridges and is likely a saddle."""
            rt_i = [rt for _, rt in ridge_i]
            mz_i = [mz for mz, _ in ridge_i]
            mean_rt_i = np.mean(rt_i)
            mean_mz_i = np.mean(mz_i)
            max_i = max(self.grid[mz, rt] for mz, rt in ridge_i)

            # Nearby ridges in RT
            neighbors = []
            for ridge_j in other_ridges:
                rt_j = [rt for _, rt in ridge_j]
                if abs(np.mean(rt_j) - mean_rt_i) < 1.0:
                    neighbors.append(ridge_j)

            # Stronger ridges
            stronger_neighbors = sorted(
                [r for r in neighbors if max(self.grid[mz, rt] for mz, rt in r) > 1.5 * max_i],
                key=lambda r: max(self.grid[mz, rt] for mz, rt in r),
                reverse=True
            )

            if len(stronger_neighbors) >= 2:
                r1, r2 = stronger_neighbors[0], stronger_neighbors[1]
                mz_r1 = np.mean([mz for mz, _ in r1])
                mz_r2 = np.mean([mz for mz, _ in r2])

                if min(mz_r1, mz_r2) < mean_mz_i < max(mz_r1, mz_r2):
                    valley_info = self._valley_score_between_ridges(r1, r2)
                    if valley_info is None or valley_info["confidence"] < 0.3:
                        return True  # weak valley between strong flanks = likely a saddle

            return False


        filtered_scores = []
        for idx, ridge in ridge_scores:
            others = [r for j, r in ridge_scores if j != idx]
            if is_mid_saddle(ridge, others):
                print(f"[FILTER] Ridge {idx} is likely a saddle between two strong peaks — ignored.")
                continue
            filtered_scores.append((idx, ridge))

        ridge_scores = filtered_scores


        overlaps = []
        for idx1, r1 in ridge_scores:
            for idx2, r2 in ridge_scores:
                if idx2 <= idx1:
                    continue
                valley = self._valley_score_between_ridges(r1, r2)
                if valley is None:
                    continue
                width_diff = abs(self._ridge_width(r1) - self._ridge_width(r2))
                smooth_diff = abs(self._intensity_smoothness(r1) - self._intensity_smoothness(r2))
                fork_sharp = self._fork_sharpness(r1, r2)
                fusion = self._fusion_score(width_diff, smooth_diff, fork_sharp)
                if fusion >= self.fusion_threshold:
                    overlaps.append({
                        "ridge_pair": (idx1, idx2),
                        "fusion_score": fusion,
                        "is_overlap": True,
                        **valley
                    })

        if overlaps:
            # === Cluster RT centers of overlapping ridges to avoid overcounting ===
            # Collect involved ridges from confirmed overlaps
            involved_ridge_indices = set()
            for o in overlaps:
                involved_ridge_indices.update(o["ridge_pair"])

            # === Deduplicate ridges based on proximity (same RT/mz center) ===
            deduped_indices = []
            used = set()

            ridge_centers = {
                i: (np.median([mz for mz, _ in self.ridges[i]]), np.median([rt for _, rt in self.ridges[i]]))
                for i in involved_ridge_indices
            }

            for i in ridge_centers:
                if i in used:
                    continue
                mz_i, rt_i = ridge_centers[i]
                cluster = [i]
                for j in ridge_centers:
                    if j == i or j in used:
                        continue
                    mz_j, rt_j = ridge_centers[j]
                    if abs(rt_i - rt_j) < 0.2 and abs(mz_i - mz_j) < 0.05:
                        cluster.append(j)
                        used.add(j)
                used.add(i)
                deduped_indices.append(i)

            # Estimate number of peaks from deduplicated set
            rt_centers = [np.mean([rt for _, rt in self.ridges[i]]) for i in deduped_indices]
            rt_centers = np.array(rt_centers).reshape(-1, 1)
            db = DBSCAN(eps=1.0, min_samples=1).fit(rt_centers)
            unique_clusters = len(set(db.labels_))
            num_peaks = max(unique_clusters, 2)  # Force minimum of 2 if overlap was detected


            return {
                "overlap_detected": len(overlaps) > 0,
                "num_overlap_events": len(overlaps),
                "num_peaks_in_overlap": num_peaks if len(overlaps) > 0 else None,
                "num_ridges_tracked": len(self.ridges),
                "overlap_details": overlaps
            }

