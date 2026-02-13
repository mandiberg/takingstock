import os
import sys
import random
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import math

# Module-level flag for debug prints; set in main() from VERBOSE
_VERBOSE = False

def _debug_print(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)


def _check_overlaps(rects: List[Tuple], label: str) -> bool:
    """
    Temporary debug helper. rects: list of (x,y,w,h) or (x,y,w,h,idx).
    Returns True if any two rects overlap. Logs to [DEBUG] [overlap-check].
    """
    def _rect(r):
        return (r[0], r[1], r[2], r[3])  # x,y,w,h
    n = len(rects)
    overlaps = []
    for i in range(n):
        ax, ay, aw, ah = _rect(rects[i])
        a_end_x, a_end_y = ax + aw, ay + ah
        for j in range(i + 1, n):
            bx, by, bw, bh = _rect(rects[j])
            b_end_x, b_end_y = bx + bw, by + bh
            if not (a_end_x <= bx or b_end_x <= ax or a_end_y <= by or b_end_y <= ay):
                overlaps.append((i, j, (ax, ay, aw, ah), (bx, by, bw, bh)))
    if overlaps:
        _debug_print(f"[DEBUG] [overlap-check] {label}: {len(overlaps)} overlap(s) found: {overlaps}")
        return True
    _debug_print(f"[DEBUG] [overlap-check] {label}: no overlaps ({n} rects)")
    return False

def chance_to_do(chance: float) -> bool:
    """
    Return True if a random number less than chance is generated.
    """
    return random.random() < chance

class BinSorter:
    def __init__(self, box_size: Tuple[int, int], size_ratios: List[Tuple],
                 min_space_threshold: int = 100,
                 nesting_layers: int = 0, nested_min_space_threshold: int = 100,
                 main_bin_fill_chance: float = 0.05,
                 item_break_scale: float = 0.0, item_break_chance: float = 0.0,
                 break_box_min_items: int = 2, break_box_max_items: int = 6,
                 break_box_fill_attempts: int = 5, break_box_coverage_threshold: float = 0.99):
        """
        Initialize the bin sorter. Items are placed until the bin is full (min_space_threshold).

        Args:
            box_size: Size of the container box in pixels (width, height)
            size_ratios: List of (width, height, weight) or (width, height, weight, expand_x, expand_y) ratios.
                         expand_x/expand_y (0-1) allow items to grow to fill gaps; omitted or 3-tuple defaults to (0, 0).
            min_space_threshold: Minimum space area required to place new items; bin is full below this
            nesting_layers: Number of nesting layers to create (0 = no nesting)
            nested_min_space_threshold: Minimum space threshold for nested bins
            main_bin_fill_chance: Probability (0–1) that the main bin's first item may use the ratio that fills
                                 the entire box. E.g. 0.05 = 5% allow full-bin ratio, 95% exclude it.
            item_break_scale: Fraction (0–1) of the bin's area. If an item's area >= this fraction of the bin, it may be "broken" into a nested bin (see item_break_chance). E.g. 0.25 = 25%. 0 = disabled.
            item_break_chance: When item area >= item_break_scale * bin_area, probability (0–1) to break it into a nested bin filled with a random subset of size_ratios (excluding the item's ratio).
            break_box_min_items: Minimum number of items to place in a break box (default 2)
            break_box_max_items: Maximum number of items to place in a break box (default 6)
            break_box_fill_attempts: Number of attempts to fill break box; accept first that yields valid placements (default 5)
            break_box_coverage_threshold: Minimum fractional coverage (0-1) for fallback layouts; reject if covered area < threshold (default 0.99)
        """
        self.box_width, self.box_height = box_size
        self.size_ratios = size_ratios
        self.min_space_threshold = min_space_threshold
        self.nesting_layers = nesting_layers
        self.nested_min_space_threshold = nested_min_space_threshold
        self.main_bin_fill_chance = main_bin_fill_chance
        self.item_break_scale = item_break_scale
        self.item_break_chance = item_break_chance
        self.break_box_min_items = break_box_min_items
        self.break_box_max_items = break_box_max_items
        self.break_box_fill_attempts = break_box_fill_attempts
        self.break_box_coverage_threshold = break_box_coverage_threshold
        self.bins = []
        self.nested_bins = {}  # Maps (bin_idx, item_idx) -> nested bin data

        # Normalize weights (convert to cumulative probabilities for efficient selection)
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights and create cumulative distribution for weighted selection."""
        if not self.size_ratios:
            self._cumulative_weights = []
            return
        
        # Extract weights, handling both (x, y) and (x, y, weight) formats for backward compatibility
        weights = []
        for ratio in self.size_ratios:
            if len(ratio) == 3:
                weights.append(ratio[2])
            else:
                weights.append(1.0)  # Default weight if not specified
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight == 0:
            # If all weights are 0, use equal weights
            weights = [1.0 / len(weights)] * len(weights)
            total_weight = 1.0
        
        # Create cumulative distribution
        self._cumulative_weights = []
        cumulative = 0.0
        for weight in weights:
            cumulative += weight / total_weight
            self._cumulative_weights.append(cumulative)
    
    def _select_weighted_ratio(self) -> Tuple[int, int]:
        """Select a ratio based on weights."""
        if not self.size_ratios:
            return None
        
        r = random.random()
        for i, cum_weight in enumerate(self._cumulative_weights):
            if r <= cum_weight:
                ratio = self.size_ratios[i]
                # Return (width_ratio, height_ratio)
                return (ratio[0], ratio[1])
        
        # Fallback to last ratio (shouldn't happen due to normalization)
        ratio = self.size_ratios[-1]
        return (ratio[0], ratio[1])

    def _get_expand_allowances(self, wr: int, hr: int) -> Tuple[float, float]:
        """Return (expand_x, expand_y) for the ratio (wr, hr) from size_ratios. Default (0.0, 0.0) if not found or 3-tuple."""
        for ratio in self.size_ratios:
            if (ratio[0], ratio[1]) == (wr, hr) and len(ratio) >= 5:
                return (float(ratio[3]), float(ratio[4]))
        return (0.0, 0.0)

    def _gaps_for_item(self, bin_items: List[Tuple], item: Tuple[int, int, int, int, int],
                       box_w: int, box_h: int, tolerance: int = 1) -> Tuple[int, int, int, int]:
        """
        For one item (x, y, w, h, idx), return (left_gap, right_gap, top_gap, bottom_gap):
        distance to bin edge or to nearest touching item on that side. Gap <= tolerance is treated as 0.
        """
        x, y, w, h, idx = item
        left_gap = x
        right_gap = box_w - (x + w)
        bottom_gap = y
        top_gap = box_h - (y + h)

        def vertical_overlap(oy: int, oh: int) -> bool:
            return not (y + h <= oy or oy + oh <= y)

        def horizontal_overlap(ox: int, ow: int) -> bool:
            return not (x + w <= ox or ox + ow <= x)

        for ox, oy, ow, oh, oidx in bin_items:
            if oidx == idx:
                continue
            if vertical_overlap(oy, oh):
                if ox + ow <= x:
                    left_gap = min(left_gap, x - (ox + ow))
                if ox >= x + w:
                    right_gap = min(right_gap, ox - (x + w))
            if horizontal_overlap(ox, ow):
                if oy + oh <= y:
                    bottom_gap = min(bottom_gap, y - (oy + oh))
                if oy >= y + h:
                    top_gap = min(top_gap, oy - (y + h))

        def clamp(g: int) -> int:
            return 0 if g <= tolerance else g

        return (clamp(left_gap), clamp(right_gap), clamp(top_gap), clamp(bottom_gap))

    def _touching_check(self, bin_items: List[Tuple], box_w: int, box_h: int, tolerance: int = 1) -> bool:
        """True if every item has all four sides touching the bin or another item (gap <= tolerance)."""
        for item in bin_items:
            left_g, right_g, top_g, bottom_g = self._gaps_for_item(bin_items, item, box_w, box_h, tolerance)
            if left_g or right_g or top_g or bottom_g:
                return False
        return True

    def _gap_fill_pass(self, bin_items: List[Tuple], box_w: int, box_h: int) -> List[Tuple]:
        """
        Expand items into gaps within their ratio's expand_x/expand_y allowances.
        Process items in (y, x) order so updated rects are used for subsequent gap checks.
        Returns new list of (x, y, w, h, item_idx).
        """
        if not bin_items:
            return list(bin_items)
        # Work on a mutable copy; process by (y, x) for deterministic order
        items_list = [list(it) for it in bin_items]
        sorted_indices = sorted(range(len(items_list)), key=lambda i: (items_list[i][1], items_list[i][0]))

        for i in sorted_indices:
            x, y, w, h, idx = items_list[i]
            base_w, base_h = w, h
            left_g, right_g, top_g, bottom_g = self._gaps_for_item(
                [tuple(it) for it in items_list], (x, y, w, h, idx), box_w, box_h
            )
            ratio_wh = self._get_item_ratio(w, h)
            expand_x, expand_y = (0.0, 0.0) if ratio_wh is None else self._get_expand_allowances(ratio_wh[0], ratio_wh[1])
            if expand_x <= 0 and expand_y <= 0:
                continue
            dw_left = min(left_g, int(base_w * expand_x))
            dw_right = min(right_g, int(base_w * expand_x))
            dh_bottom = min(bottom_g, int(base_h * expand_y))
            dh_top = min(top_g, int(base_h * expand_y))
            new_x = x - dw_left
            new_y = y - dh_top
            new_w = w + dw_left + dw_right
            new_h = h + dh_bottom + dh_top
            # Clamp to bin
            if new_x < 0:
                new_w += new_x
                new_x = 0
            if new_y < 0:
                new_h += new_y
                new_y = 0
            if new_x + new_w > box_w:
                new_w = box_w - new_x
            if new_y + new_h > box_h:
                new_h = box_h - new_y
            if new_w >= 1 and new_h >= 1:
                if (new_x, new_y, new_w, new_h) != (x, y, w, h):
                    _debug_print(f"[DEBUG] [gap-fill] item idx={idx} changed from ({x},{y}) {w}x{h} -> ({new_x},{new_y}) {new_w}x{new_h}")
                items_list[i] = [new_x, new_y, new_w, new_h, idx]

        return [tuple(it) for it in items_list]

    def _get_random_restricted_ratios(self, parent_w_ratio: int, parent_h_ratio: int) -> List[Tuple]:
        """Return a random subset of size_ratios for filling a 'break' nested bin, excluding the parent ratio when possible."""
        other = [r for r in self.size_ratios if (r[0], r[1]) != (parent_w_ratio, parent_h_ratio)]
        pool = other if other else list(self.size_ratios)
        if len(pool) <= 2:
            return pool
        k = random.randint(2, min(4, len(pool)))
        return random.sample(pool, k)

    def _coverage_fraction(self, placements: List[Tuple[int, int, int, int]],
                           box_w: int, box_h: int) -> float:
        """Return sum of rect areas / box area, capped at 1.0. Used to reject fallback layouts with gaps."""
        if not placements or box_w <= 0 or box_h <= 0:
            return 0.0
        covered = sum(r[2] * r[3] for r in placements)
        return min(1.0, covered / (box_w * box_h))

    def _try_layout_n_items(self, box_w: int, box_h: int, n: int,
                            ratios_only: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Try to place n items in a layout that tiles the box. Returns placements or [] if impossible.
        Supports layouts: 2=1x2 or 2x1, 3=1x3 or 3x1, 4=2x2, 5=5x1 or 1x5, 6=2x3 or 3x2.
        """
        if n < 2 or n > 6 or len(ratios_only) < n:
            return []
        
        W, H = float(box_w), float(box_h)
        target_aspect = W / H
        
        # Try different layout arrangements based on n
        layouts_to_try = []
        if n == 2:
            layouts_to_try = [(1, 2), (2, 1)]  # 1 row x 2 cols, or 2 rows x 1 col
        elif n == 3:
            layouts_to_try = [(1, 3), (3, 1)]
        elif n == 4:
            layouts_to_try = [(2, 2)]
        elif n == 5:
            layouts_to_try = [(1, 5), (5, 1)]
        elif n == 6:
            layouts_to_try = [(2, 3), (3, 2)]
        
        for rows, cols in layouts_to_try:
            for attempt in range(50):  # Try up to 50 random ratio combinations
                if len(ratios_only) >= n:
                    selected = random.sample(ratios_only, n)
                else:
                    selected = [random.choice(ratios_only) for _ in range(n)]
                
                # Arrange ratios into rows x cols grid
                ratio_grid = []
                for r in range(rows):
                    row_start = r * cols
                    ratio_grid.append(selected[row_start:row_start + cols])
                
                # Compute row heights so each row has width W
                row_heights = []
                for row_idx, row_ratios in enumerate(ratio_grid):
                    row_width_rate = sum(wr / hr for wr, hr in row_ratios if hr > 0)
                    if row_width_rate <= 0:
                        break
                    row_height = W / row_width_rate
                    row_heights.append(row_height)
                else:
                    # All rows computed successfully
                    total_height = sum(row_heights)
                    if total_height <= 0:
                        continue
                    
                    # Scale to fit box_h
                    scale_h = H / total_height
                    row_heights = [h * scale_h for h in row_heights]
                    
                    # Convert to integers and compute item dimensions
                    row_heights_int = [max(1, int(round(h))) for h in row_heights]
                    if sum(row_heights_int) != box_h:
                        # Adjust last row
                        row_heights_int[-1] = max(1, box_h - sum(row_heights_int[:-1]))
                    
                    placements = []
                    y = 0
                    valid = True
                    
                    for row_idx, (row_ratios, row_h) in enumerate(zip(ratio_grid, row_heights_int)):
                        # Compute widths for this row preserving ratios
                        item_widths = []
                        for wr, hr in row_ratios:
                            if hr <= 0:
                                valid = False
                                break
                            w = max(1, int(round(row_h * wr / hr)))
                            item_widths.append(w)
                        if not valid:
                            break
                        
                        # Scale to fit box_w proportionally (preserves ratios)
                        row_sum = sum(item_widths)
                        if row_sum > 0 and row_sum != box_w:
                            scale_w = box_w / row_sum
                            item_widths = [max(1, int(round(w * scale_w))) for w in item_widths]
                            # If still not exact, scale again proportionally
                            if sum(item_widths) != box_w and sum(item_widths) > 0:
                                scale_w2 = box_w / sum(item_widths)
                                item_widths = [max(1, int(round(w * scale_w2))) for w in item_widths]
                                # Only adjust last item if error is tiny (1-2 pixels)
                                if abs(sum(item_widths) - box_w) <= 2:
                                    item_widths[-1] = max(1, box_w - sum(item_widths[:-1]))
                        
                        # Recalculate widths from exact row_h to ensure all items have same height
                        # This prevents overlaps while preserving aspect ratios as much as possible
                        item_widths_from_height = []
                        for wr, hr in row_ratios:
                            if hr <= 0:
                                valid = False
                                break
                            w = max(1, int(round(row_h * wr / hr)))
                            item_widths_from_height.append(w)
                        if not valid:
                            break
                        
                        # Scale these widths to fit box_w proportionally (preserves ratios)
                        row_sum_h = sum(item_widths_from_height)
                        if row_sum_h > 0 and row_sum_h != box_w:
                            scale_w_h = box_w / row_sum_h
                            item_widths_from_height = [max(1, int(round(w * scale_w_h))) for w in item_widths_from_height]
                            if sum(item_widths_from_height) != box_w and sum(item_widths_from_height) > 0:
                                scale_w_h2 = box_w / sum(item_widths_from_height)
                                item_widths_from_height = [max(1, int(round(w * scale_w_h2))) for w in item_widths_from_height]
                                if abs(sum(item_widths_from_height) - box_w) <= 2:
                                    item_widths_from_height[-1] = max(1, box_w - sum(item_widths_from_height[:-1]))
                        
                        # Place items in this row - all with exact height row_h
                        x = 0
                        for (wr, hr), w in zip(row_ratios, item_widths_from_height):
                            h = row_h  # All items in row have same height to prevent overlaps
                            # Verify item fits
                            if x + w > box_w or y + h > box_h:
                                valid = False
                                break
                            placements.append((x, y, w, h))
                            x += w
                        
                        if not valid:
                            break
                        # All items in row have height row_h, so increment by row_h
                        y += row_h
                    
                    if valid and len(placements) == n:
                        # Verify aspect ratios, bounds, and row/column sums
                        y_positions = sorted(set(y for _, y, _, _ in placements))
                        prev_y_end = -1
                        total_height = 0
                        for y_pos in y_positions:
                            row_items = [(x, w, h) for x, y, w, h in placements if y == y_pos]
                            row_items.sort()
                            row_sum = sum(w for _, w, _ in row_items)
                            row_h = row_items[0][2] if row_items else 0
                            # Check row sums to box_w (allow 1-2 pixel rounding)
                            if abs(row_sum - box_w) > 2:
                                valid = False
                                break
                            # Check rows don't overlap
                            if y_pos < prev_y_end:
                                valid = False
                                break
                            prev_y_end = y_pos + row_h
                            total_height = max(total_height, y_pos + row_h)
                        
                        # Check total height equals box_h
                        if valid and abs(total_height - box_h) > 2:
                            valid = False
                        
                        # Verify all items fit and have correct aspect ratios
                        if valid:
                            for (x, y, w, h), (wr, hr) in zip(placements, selected):
                                if h == 0 or hr == 0 or x + w > box_w or y + h > box_h:
                                    valid = False
                                    break
                                aspect_curr = w / h
                                aspect_target = wr / hr
                                if abs(aspect_curr - aspect_target) / aspect_target > 0.02:
                                    valid = False
                                    break
                        
                        if valid:
                            return placements
        
        return []  # No valid layout found

    def _fix_overlaps(self, placements: List[Tuple[int, int, int, int]], 
                     box_w: int, box_h: int) -> List[Tuple[int, int, int, int]]:
        """
        Fix overlaps in placements by adjusting y-positions. For items that overlap horizontally,
        ensures they don't overlap vertically by moving overlapping items down.
        """
        if not placements:
            return placements
        
        # Sort items by y-position, then x-position
        sorted_placements = sorted(placements, key=lambda item: (item[1], item[0]))
        fixed_placements = []
        
        for x, y, w, h in sorted_placements:
            original_y = y
            # Check for overlaps with already-placed items
            max_y_end = 0
            overlapping_items = []
            for fx, fy, fw, fh in fixed_placements:
                # Check if items overlap horizontally
                if not (x + w <= fx or fx + fw <= x):
                    # They overlap horizontally - ensure this item starts after the previous one ends
                    item_y_end = fy + fh
                    max_y_end = max(max_y_end, item_y_end)
                    overlapping_items.append((fx, fy, fw, fh))
            
            # Move this item down if it would overlap
            if y < max_y_end:
                y = max_y_end
                _debug_print(f"[DEBUG] [fix-overlaps] Moved item ({x},{original_y}) {w}x{h} down to y={y} (max_y_end={max_y_end}, overlapping with {len(overlapping_items)} items)")
            
            # Ensure item doesn't exceed box bounds
            if y + h > box_h:
                # Item would exceed box - shrink height to fit
                original_h = h
                h = max(1, box_h - y)
                _debug_print(f"[DEBUG] [fix-overlaps] Shrunk item ({x},{y}) height from {original_h} to {h} to fit in box_h={box_h}")
                if h <= 0:
                    _debug_print(f"[DEBUG] [fix-overlaps] Skipping item ({x},{y}) {w}x{h} - cannot fit in box")
                    continue  # Skip this item if it can't fit
            
            fixed_placements.append((x, y, w, h))
        
        return fixed_placements

    def _can_place_in_bin(self, bin_items: List[Tuple], item_width: int, 
                          item_height: int, bin_width: int, bin_height: int) -> bool:
        """Check if item can fit in bin (simple check without exact positioning)."""
        # Simple check: see if there's enough total space
        used_area = sum(w * h for _, _, w, h, _ in bin_items)
        available_area = bin_width * bin_height - used_area
        return available_area >= item_width * item_height
    
    def _find_position(self, bin_items: List[Tuple], item_width: int, item_height: int,
                      bin_width: int, bin_height: int) -> Tuple[int, int]:
        """
        Find a position for the item using a simple bottom-left fill approach.
        Returns (x, y) or (None, None) if no position found.
        """
        if not bin_items:
            # Randomly place first item in one of the four corners
            corners = [
                (0, 0),  # Bottom-left
                (bin_width - item_width, 0),  # Bottom-right
                (0, bin_height - item_height),  # Top-left
                (bin_width - item_width, bin_height - item_height)  # Top-right
            ]
            # Filter corners where item fits
            valid_corners = [(x, y) for x, y in corners 
                           if 0 <= x and x + item_width <= bin_width and 
                              0 <= y and y + item_height <= bin_height]
            if valid_corners:
                return random.choice(valid_corners)
            # Fallback to (0, 0) if somehow no corner works
            return (0, 0)
        
        # Collect all valid candidate positions
        candidates = []
        
        # Strategy 1: Try left edge positions (fill gaps on left side)
        # Find all y positions where we can place on the left edge
        min_x = min(x for x, _, _, _, _ in bin_items) if bin_items else 0
        if min_x > 0:
            # There's space on the left - try filling it
            for test_y in range(0, bin_height - item_height + 1, max(1, item_height // 4)):
                test_x = 0
                if not self._overlaps((test_x, test_y, item_width, item_height), bin_items):
                    if test_x + item_width <= bin_width and test_y + item_height <= bin_height:
                        # Score: prefer lower y and leftmost x
                        score = -test_y * 1000 - test_x
                        candidates.append((test_x, test_y, score))
        
        # Strategy 2: Try positions to the right of existing items
        sorted_items = sorted(bin_items, key=lambda item: (item[1], item[0]))
        for x, y, w, h, _ in sorted_items:
            # Try right of this item
            new_x = x + w
            if new_x + item_width <= bin_width and y + item_height <= bin_height:
                if not self._overlaps((new_x, y, item_width, item_height), bin_items):
                    # Score: prefer positions closer to left and bottom
                    score = -new_x * 100 - y
                    candidates.append((new_x, y, score))
            
            # Try above this item
            new_y = y + h
            if new_y + item_height <= bin_height and x + item_width <= bin_width:
                if not self._overlaps((x, new_y, item_width, item_height), bin_items):
                    # Score: prefer positions closer to left and bottom
                    score = -x * 100 - new_y
                    candidates.append((x, new_y, score))
        
        # Strategy 3: Try bottom-left fill (original strategy)
        max_y = max(y + h for _, y, _, h, _ in bin_items) if bin_items else 0
        if max_y + item_height <= bin_height:
            test_x = 0
            test_y = max_y
            if not self._overlaps((test_x, test_y, item_width, item_height), bin_items):
                score = -test_x * 1000 - test_y
                candidates.append((test_x, test_y, score))
        
        # Strategy 4: Try filling gaps between items (scan for empty rectangles)
        # Check positions along the left edge and between items
        for test_y in range(0, bin_height - item_height + 1, max(1, item_height // 4)):
            for test_x in range(0, bin_width - item_width + 1, max(1, item_width // 4)):
                if not self._overlaps((test_x, test_y, item_width, item_height), bin_items):
                    if test_x + item_width <= bin_width and test_y + item_height <= bin_height:
                        # Score: prefer leftmost and bottommost positions
                        score = -test_x * 100 - test_y
                        candidates.append((test_x, test_y, score))
        
        # Sort candidates by score (best first) and return the best valid position
        if candidates:
            candidates.sort(key=lambda c: c[2], reverse=True)
            return (candidates[0][0], candidates[0][1])
        
        return (None, None)
    
    def _overlaps(self, new_item: Tuple[int, int, int, int], 
                  existing_items: List[Tuple]) -> bool:
        """Check if new_item overlaps with any existing items."""
        new_x, new_y, new_w, new_h = new_item
        for x, y, w, h, _ in existing_items:
            if not (new_x + new_w <= x or x + w <= new_x or 
                   new_y + new_h <= y or y + h <= new_y):
                return True
        return False
    
    def _get_remaining_space(self, bin_items: List[Tuple]) -> int:
        """Calculate the remaining space area in a bin."""
        used_area = sum(w * h for _, _, w, h, _ in bin_items)
        total_area = self.box_width * self.box_height
        return total_area - used_area
    
    def _get_largest_fittable_rectangle(self, bin_items: List[Tuple]) -> int:
        """
        Find the largest rectangle that can fit in the remaining space.
        Returns the area of the largest fittable rectangle.
        This is used as the "full enough" metric in infinite mode (fragmentation-aware).
        """
        candidate = self._find_largest_placeable_item(bin_items)
        return candidate[2] * candidate[3] if candidate is not None else 0
    
    def _is_position_free(self, x: int, y: int, bin_items: List[Tuple]) -> bool:
        """Check if a single point (x, y) is free of items."""
        for item_x, item_y, item_w, item_h, _ in bin_items:
            if item_x <= x < item_x + item_w and item_y <= y < item_y + item_h:
                return False
        return True

    def _find_largest_placeable_item(self, bin_items: List[Tuple], 
                                    exclude_ratio_for_first: bool = False) -> Tuple[int, int, int, int, Tuple[int, int]]:
        """
        Find the best rectangle that can be placed into the given bin, using
        the available `size_ratios` (including rotation). Prioritizes weights over pure area.
        
        Strategy:
        - For each ratio, start at the maximum scale allowed by the bin dimensions
          (so the item touches either bin width or bin height), then scale down until
          it can be positioned without overlaps.
        - Calculate weighted score = area * weight for each candidate
        - Pick the candidate with the highest weighted score (prioritizes weights).
        
        Args:
            exclude_ratio_for_first: If True, exclude the ratio stored in _exclude_ratio_for_first_item
        
        Returns:
            (x, y, w, h, (w_ratio, h_ratio)) or None if nothing fits.
        """
        if not self.size_ratios:
            return None
        
        # Get ratio to exclude for first item (if any)
        exclude_ratio = None
        if exclude_ratio_for_first and hasattr(self, '_exclude_ratio_for_first_item'):
            exclude_ratio = self._exclude_ratio_for_first_item
            if exclude_ratio:
                _debug_print(f"[DEBUG] [exclude-parent] _find_largest_placeable_item: bin box_size=({self.box_width},{self.box_height}), exclude_ratio_for_first=True, _exclude_ratio_for_first_item={exclude_ratio}")

        # Scale-down schedule: start at "as big as the bin allows", then back off.
        # More values = better fit quality, slower. This is a reasonable compromise.
        shrink_factors = [1.0, 0.98, 0.95, 0.92, 0.9, 0.87, 0.85, 0.82, 0.8, 0.77, 0.75,
                          0.72, 0.7, 0.67, 0.65, 0.62, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35,
                          0.3, 0.25, 0.2, 0.15, 0.1]

        best = None  # (x, y, w, h, (w_ratio, h_ratio), weight)
        best_score = 0  # Combined score of area * weight
        
        # Build weighted list of ratios to try
        weighted_ratios = []
        for ratio in self.size_ratios:
            width_ratio, height_ratio = ratio[0], ratio[1]
            weight = ratio[2] if len(ratio) >= 3 else 1.0
            weighted_ratios.append((width_ratio, height_ratio, weight))
        
        # Sort by weight (descending) so we try higher-weighted ratios first
        weighted_ratios.sort(key=lambda x: x[2], reverse=True)
        
        # Collect all candidates first, then select best by weighted score
        candidates = []
        
        for width_ratio, height_ratio, weight in weighted_ratios:
            # Skip excluded ratio if this is the first item
            if exclude_ratio:
                parent_w, parent_h = exclude_ratio
                # Main bin: exclude by aspect ratio (ratio that fills box). Nested: exact match.
                if (self.box_width, self.box_height) == (parent_w, parent_h):
                    if width_ratio * parent_h == height_ratio * parent_w:
                        _debug_print(f"[DEBUG] [exclude-parent] Skipping ratio ({width_ratio},{height_ratio}) [main bin box-fill aspect, box=({self.box_width},{self.box_height})]")
                        continue
                elif (width_ratio == parent_w and height_ratio == parent_h):
                    _debug_print(f"[DEBUG] [exclude-parent] Skipping ratio ({width_ratio},{height_ratio}) [nested bin: exact match to parent {exclude_ratio}, box=({self.box_width},{self.box_height})]")
                    continue
            
            # Use only the original orientation (no rotation)
            w_ratio, h_ratio = width_ratio, height_ratio
            
            if w_ratio <= 0 or h_ratio <= 0:
                continue

            # Max scale such that width<=box_width and height<=box_height.
            max_scale = min(self.box_width / w_ratio, self.box_height / h_ratio)
            if max_scale <= 0:
                continue

            for f in shrink_factors:
                w = int(w_ratio * max_scale * f)
                h = int(h_ratio * max_scale * f)
                if w < 1 or h < 1:
                    continue

                # Second item in bin: largest side of item must not exceed smallest side of bin
                if len(bin_items) == 1:
                    bin_smallest_side = min(self.box_width, self.box_height)
                    if max(w, h) > bin_smallest_side:
                        continue

                area = w * h

                if not self._can_place_in_bin(bin_items, w, h, self.box_width, self.box_height):
                    continue
                x, y = self._find_position(bin_items, w, h, self.box_width, self.box_height)
                if x is None:
                    continue

                # Calculate weighted score: area * weight
                # This prioritizes both size and weight preference
                score = area * weight
                candidates.append((x, y, w, h, (w_ratio, h_ratio), score, area, weight))
                
                # We found the largest f that fits (since we iterate down).
                break
        
        # Select candidate with highest weighted score
        if candidates:
            # Sort by score (descending), then by area as tiebreaker
            candidates.sort(key=lambda c: (c[5], c[6]), reverse=True)
            best_candidate = candidates[0]
            best = (best_candidate[0], best_candidate[1], best_candidate[2], 
                   best_candidate[3], best_candidate[4])
            # Debug ratio selection only for first item or when excluding parent ratio
            if (hasattr(self, '_exclude_ratio_for_first_item') and self._exclude_ratio_for_first_item) or len(bin_items) == 0:
                sel = best_candidate[4]
                msg = f"[DEBUG] Selected ratio {sel} with area={best_candidate[6]}, weight={best_candidate[7]}, score={best_candidate[5]:.0f}"
                # Only label as nested when exclude_ratio != box size (main bin uses box size as exclude for box-fill)
                if exclude_ratio and (self.box_width, self.box_height) != exclude_ratio:
                    msg += f" [exclude-parent: first item in nested bin box=({self.box_width},{self.box_height}), excluded parent ratio was {exclude_ratio}]"
                print(msg)

        return best
    
    def _can_fit_any_item(self, bin_items: List[Tuple], scale: float) -> bool:
        """Check if any item from size_ratios can fit in the bin with given scale."""
        for ratio in self.size_ratios:
            width_ratio, height_ratio = ratio[0], ratio[1]
            width = int(width_ratio * scale)
            height = int(height_ratio * scale)
            if width < 1 or height < 1:
                continue
            if self._can_place_in_bin(bin_items, width, height, self.box_width, self.box_height):
                x, y = self._find_position(bin_items, width, height, self.box_width, self.box_height)
                if x is not None:
                    return True
        return False
    
    def sort(self, nested_start_item_idx: int = None) -> List[List[Tuple]]:
        """Run the bin sorting algorithm: fill one bin until min_space_threshold, then create nested bins if nesting_layers > 0.

        Args:
            nested_start_item_idx: Starting item index for nested bins (used when this is a nested sorter)
        """
        _debug_print(f"\n[DEBUG] sort() called: nesting_layers={self.nesting_layers}")
        self.bins = self._sort_infinite()

        if self.nesting_layers > 0:
            _debug_print(f"[DEBUG] Initial sorting complete: {len(self.bins)} bins created")
            for i, bin_items in enumerate(self.bins):
                _debug_print(f"  Bin {i}: {len(bin_items)} items")

        if self.nesting_layers > 0:
            _debug_print(f"[DEBUG] Creating nested bins with {self.nesting_layers} layers")
            if nested_start_item_idx is not None:
                self._create_nested_bins(self.nesting_layers, nested_start_item_idx)
            else:
                self._create_nested_bins(self.nesting_layers)

        # Gap-fill pass: expand items into gaps within their expand_x/expand_y allowances
        for i, bin_items in enumerate(self.bins):
            self.bins[i] = self._gap_fill_pass(bin_items, self.box_width, self.box_height)
        for bin_key, nested_data in self.nested_bins.items():
            pw, ph = nested_data['parent_pos'][2], nested_data['parent_pos'][3]
            nested_items = nested_data['items']
            _check_overlaps(nested_items, f"nested bin {bin_key} BEFORE gap fill")
            # Apply gap filling
            nested_items = self._gap_fill_pass(nested_items, pw, ph)
            _check_overlaps(nested_items, f"nested bin {bin_key} AFTER gap fill (before overlap fix)")
            # Fix any overlaps that gap filling may have created
            # Convert to (x, y, w, h) format for _fix_overlaps
            placements = [(x, y, w, h) for (x, y, w, h, _) in nested_items]
            fixed_placements = self._fix_overlaps(placements, pw, ph)
            # Convert back to (x, y, w, h, idx) format
            nested_data['items'] = [(x, y, w, h, idx) for (x, y, w, h), (_, _, _, _, idx) in zip(fixed_placements, nested_items)]
            _check_overlaps(nested_data['items'], f"nested bin {bin_key} AFTER overlap fix")

        # Final overlap check on exact data we're returning (catches any missed path)
        for bin_key, nested_data in self.nested_bins.items():
            _check_overlaps(nested_data['items'], f"FINAL nested bin {bin_key} (data used for draw)")
        _debug_print(f"[DEBUG] Final nested_bins count: {len(self.nested_bins)} [box={self.box_width}x{self.box_height}]")
        return self.bins
    
    def _count_nested_items(self, nested_bins_dict: dict) -> int:
        """Recursively count all items in nested bins.
        
        Args:
            nested_bins_dict: Dictionary of nested bins (can be self.nested_bins or nested_bins from nested_data)
        
        Returns:
            Total count of items in all nested bins
        """
        total = 0
        for nested_data in nested_bins_dict.values():
            # Count items in this nested bin
            nested_items = nested_data.get('items', [])
            total += len(nested_items)
            
            # Recursively count items in deeper nested bins
            deeper_nested_bins = nested_data.get('nested_bins', {})
            if deeper_nested_bins:
                total += self._count_nested_items(deeper_nested_bins)
        
        return total
    
    def _count_nested_bins(self, nested_bins_dict: dict) -> int:
        """Recursively count all nested bins.
        
        Args:
            nested_bins_dict: Dictionary of nested bins (can be self.nested_bins or nested_bins from nested_data)
        
        Returns:
            Total count of nested bins (including deeply nested ones)
        """
        total = 0
        for nested_data in nested_bins_dict.values():
            # Count this nested bin
            total += 1
            
            # Recursively count deeper nested bins
            deeper_nested_bins = nested_data.get('nested_bins', {})
            if deeper_nested_bins:
                total += self._count_nested_bins(deeper_nested_bins)
        
        return total
    
    def get_total_items_count(self) -> int:
        """Get the total count of all items including nested items.
        
        Returns:
            Total count of items in main bins + all nested bins
        """
        # Count items in main bins
        main_items = sum(len(bin_items) for bin_items in self.bins)
        
        # Count items in nested bins (recursively)
        nested_items = self._count_nested_items(self.nested_bins)
        
        return main_items + nested_items
    
    def get_total_bins_count(self) -> int:
        """Get the total count of all bins including nested bins.
        
        Returns:
            Total count of main bins + all nested bins
        """
        
        # Count nested bins (recursively)
        nested_bins_count = self._count_nested_bins(self.nested_bins)
        
        return nested_bins_count + len(self.bins)
    
    def _create_nested_bin_items_fit_width(self, nested_bin_width: int, nested_bin_height: int, 
                                          target_num_items: int, scale_factor: float = 1.0,
                                          use_specific_ratios: List[Tuple[int, int]] = None) -> List[Tuple[float, float, int]]:
        """
        Create items for nested bins, sized to fit the nested bin's width.
        Items are scaled so their width matches the nested bin width (or fits within it).
        
        Args:
            nested_bin_width: Width of the nested bin
            nested_bin_height: Height of the nested bin
            target_num_items: Number of items to create
            scale_factor: Additional scale factor to apply (for retries)
            use_specific_ratios: If provided, use these ratios instead of weighted selection
        """
        if not self.size_ratios:
            return []
        
        items = []
        
        # Determine which ratios to use
        if use_specific_ratios:
            ratios_to_use = use_specific_ratios
        else:
            # Use weighted selection for each item
            ratios_to_use = None
        
        for idx in range(target_num_items):
            # Select ratio
            if use_specific_ratios and idx < len(use_specific_ratios):
                ratio = use_specific_ratios[idx]
                # Handle both (w, h) and (w, h, weight) formats
                if len(ratio) >= 2:
                    width_ratio, height_ratio = ratio[0], ratio[1]
                else:
                    width_ratio, height_ratio = self._select_weighted_ratio()
            else:
                width_ratio, height_ratio = self._select_weighted_ratio()
            
            # Scale to match nested bin width exactly (primary constraint)
            # Calculate scale so width matches nested_bin_width exactly
            scale_by_width = nested_bin_width / width_ratio if width_ratio > 0 else 0
            
            # Use width scale exactly - prioritize width filling
            # Height may exceed bin height, but packing algorithm will handle it
            scale = scale_by_width * scale_factor
            
            width = int(width_ratio * scale)
            height = int(height_ratio * scale)
            
            # Debug output removed for cleaner logs
            
            # Ensure minimum size
            if width < 1:
                width = 1
            if height < 1:
                height = 1
            
            items.append((width, height, idx))
        
        return items
    
    def _get_item_ratio(self, width: int, height: int) -> Tuple[int, int]:
        """Determine the ratio of an item by finding the closest matching ratio from size_ratios."""
        if not self.size_ratios:
            return None
        
        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        
        best_match = None
        best_diff = float('inf')
        
        for ratio in self.size_ratios:
            w_ratio, h_ratio = ratio[0], ratio[1]
            
            # Check only original orientation (no rotation)
            ratio_aspect = w_ratio / h_ratio if h_ratio > 0 else 0
            diff = abs(aspect_ratio - ratio_aspect)
            if diff < best_diff:
                best_diff = diff
                best_match = (w_ratio, h_ratio)
        
        return best_match
    
    def _remap_nested_bin_data(self, nested_bins_dict: dict, offset: int, bin_idx: int, parent_item_idx: int, parent_path: tuple = (), remapped_items_list: List[Tuple] = None) -> dict:
        """Recursively remap nested bin keys and item indices.
        
        Args:
            nested_bins_dict: Dictionary of nested bins to remap
            offset: Offset to add to item indices
            bin_idx: Bin index for the parent
            parent_item_idx: Parent item index
            parent_path: Path tuple for deeper nesting
            remapped_items_list: List of remapped items (x, y, w, h, idx) to map old_item_idx to correct remapped index
        
        Returns:
            Remapped nested bins dictionary
        """
        remapped = {}
        for (nested_bin_idx, old_item_idx), nested_data in nested_bins_dict.items():
            # Remap the item index in the key
            # If remapped_items_list is provided, old_item_idx is the original index from nested_sorter.bins[0]
            # We need to find the corresponding remapped index in remapped_items_list
            # Otherwise, assume old_item_idx is an original index and add offset
            if remapped_items_list is not None:
                # old_item_idx can be either (a) original position 0,1,2,... in the creator's bins[0],
                # or (b) already a remapped index from a later recursive _create_nested_bins.
                # (a) Map position -> remapped index via remapped_items_list[i][4].
                # (b) If old_item_idx is already one of the indices in the list, use it as-is.
                indices_in_list = [it[4] for it in remapped_items_list]
                if 0 <= old_item_idx < len(remapped_items_list):
                    remapped_item_idx = remapped_items_list[old_item_idx][4]
                elif old_item_idx in indices_in_list:
                    remapped_item_idx = old_item_idx
                else:
                    remapped_item_idx = offset + old_item_idx
            else:
                remapped_item_idx = offset + old_item_idx
            if parent_path:
                remapped_key = parent_path + (remapped_item_idx,)
            else:
                remapped_key = (bin_idx, parent_item_idx, remapped_item_idx)
            
            # Remap item indices within nested_data['items']
            remapped_data = nested_data.copy()
            remapped_items = []
            for nx, ny, nw, nh, item_idx in nested_data.get('items', []):
                remapped_items.append((nx, ny, nw, nh, offset + item_idx))
            remapped_data['items'] = remapped_items
            
            # Recursively remap deeper nested bins; pass this level's remapped items so
            # inner old_item_idx (position in that level's bins) maps to correct remapped index
            if 'nested_bins' in nested_data and nested_data['nested_bins']:
                remapped_data['nested_bins'] = self._remap_nested_bin_data(
                    nested_data['nested_bins'], offset, bin_idx, parent_item_idx, remapped_key, remapped_items
                )
            
            remapped[remapped_key] = remapped_data
        
        return remapped
    
    def _create_nested_bins(self, remaining_layers: int, start_item_idx: int = None):
        """Recursively create nested bins inside items. Always succeeds by trying different approaches.
        
        Args:
            remaining_layers: Number of nesting layers remaining
            start_item_idx: Starting item index for nested items (None = calculate from main bins)
        """
        _debug_print(f"\n[DEBUG] _create_nested_bins called with remaining_layers={remaining_layers}, start_item_idx={start_item_idx}")
        _debug_print(f"[DEBUG] Total bins: {len(self.bins)}")
        
        # Calculate starting index if not provided
        if start_item_idx is None:
            start_item_idx = sum(len(bin_items) for bin_items in self.bins)
            _debug_print(f"[DEBUG] Calculated start_item_idx: {start_item_idx}")
        
        if remaining_layers <= 0:
            _debug_print(f"[DEBUG] No more nesting layers remaining, returning")
            return
        
        # Process each bin
        for bin_idx, bin_items in enumerate(self.bins):
            _debug_print(f"\n[DEBUG] Processing bin {bin_idx} with {len(bin_items)} items")
            
            if not bin_items:
                _debug_print(f"[DEBUG] Bin {bin_idx} is empty, skipping")
                continue
            
            # Find the largest items (up to 2)
            items_with_area = [(item, item[2] * item[3]) for item in bin_items]
            items_with_area.sort(key=lambda x: x[1], reverse=True)
            
            _debug_print(f"[DEBUG] Top items by area:")
            for i, (item, area) in enumerate(items_with_area[:2]):
                x, y, w, h, idx = item
                print(f"  {i+1}. Item {idx}: {w}x{h} (area={area})")
            
            # Filter candidates to only include items large enough to potentially fit multiple nested items
            # Check if item dimensions allow at least 2 items to fit
            candidates = []
            for item, area in items_with_area[:min(2, len(items_with_area))]:
                x, y, w, h, idx = item
                # Try to see if we can fit at least 2 items of any ratio
                can_fit_multiple = False
                for ratio in self.size_ratios:
                    w_ratio, h_ratio = ratio[0], ratio[1]
                    # Try both orientations
                    for test_w_ratio, test_h_ratio in [(w_ratio, h_ratio), (h_ratio, w_ratio)]:
                        if test_w_ratio <= 0 or test_h_ratio <= 0:
                            continue
                        # Calculate scale to fit width
                        scale_by_width = w / test_w_ratio if test_w_ratio > 0 else 0
                        # Calculate scale to fit height
                        scale_by_height = h / test_h_ratio if test_h_ratio > 0 else 0
                        # Use the smaller scale to ensure it fits
                        scale = min(scale_by_width, scale_by_height)
                        if scale <= 0:
                            continue
                        item_w = int(test_w_ratio * scale)
                        item_h = int(test_h_ratio * scale)
                        if item_w < 1 or item_h < 1:
                            continue
                        # Check if we can fit at least 2 items
                        # Simple check: if item width is less than half the bin width, or
                        # item height is less than half the bin height, we might fit 2
                        if (item_w * 2 <= w and item_h <= h) or (item_h * 2 <= h and item_w <= w):
                            can_fit_multiple = True
                            break
                    if can_fit_multiple:
                        break
                
                if can_fit_multiple:
                    candidates.append((item, area))
            
            # If no candidates can fit multiple items, try all items anyway (might still work with different ratios)
            if not candidates:
                candidates = items_with_area[:min(2, len(items_with_area))]
            
            random.shuffle(candidates)
            
            nested_bin_created = False
            for candidate_item, _ in candidates:
                x, y, w, h, item_idx = candidate_item
                _debug_print(f"[DEBUG] Trying item {item_idx}: size={w}x{h}, all bin item indices: {[ni for _, _, _, _, ni in bin_items]}")
                
                # Determine parent item's ratio (exact match only, no rotation)
                parent_ratio = self._get_item_ratio(w, h)
                if parent_ratio:
                    parent_w_ratio, parent_h_ratio = parent_ratio
                    _debug_print(f"[DEBUG] [exclude-parent] _create_nested_bins: remaining_layers={remaining_layers}, parent item {w}x{h} -> ratio ({parent_w_ratio},{parent_h_ratio}) via _get_item_ratio")
                else:
                    parent_w_ratio, parent_h_ratio = None, None
                    _debug_print(f"[DEBUG] [exclude-parent] _create_nested_bins: remaining_layers={remaining_layers}, parent item {w}x{h} -> no ratio from _get_item_ratio")
                
                # Create nested sorter (fills bin until min_space_threshold)
                # This will fill the nested bin completely using infinite mode
                # Try multiple approaches to ensure multiple items can fit:
                # 1. Try with parent ratio excluded (preferred)
                # 2. Try with parent ratio allowed (if excluding doesn't work)
                nested_bin_success = False
                scale_factors = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]  # Try progressively smaller scales
                nested_sorter = None
                num_nested_items = 0
                
                # First, try with parent ratio excluded
                exclude_parent = True
                for attempt in range(2):  # Try twice: with exclusion, then without
                    for scale_factor in scale_factors:
                        # Create nested sorter with scaled-down threshold to encourage multiple items
                        scaled_threshold = int(self.nested_min_space_threshold * scale_factor)
                        nested_sorter = BinSorter(
                            box_size=(w, h),
                            size_ratios=self.size_ratios,
                            min_space_threshold=scaled_threshold,
                            nesting_layers=remaining_layers - 1,
                            nested_min_space_threshold=self.nested_min_space_threshold
                        )
                        
                        # Store parent ratio in nested sorter so it can exclude it for first item
                        if exclude_parent and parent_w_ratio is not None and parent_h_ratio is not None:
                            nested_sorter._exclude_ratio_for_first_item = (parent_w_ratio, parent_h_ratio)
                            exclude_msg = "excluding parent ratio"
                            _debug_print(f"[DEBUG] [exclude-parent] _create_nested_bins: set nested_sorter._exclude_ratio_for_first_item=({parent_w_ratio},{parent_h_ratio}) for nested box {w}x{h}, remaining_layers={remaining_layers} -> nest depth (before sort)")
                        else:
                            nested_sorter._exclude_ratio_for_first_item = None
                            exclude_msg = "allowing parent ratio"
                            if not exclude_parent:
                                _debug_print(f"[DEBUG] [exclude-parent] _create_nested_bins: NOT setting exclude (exclude_parent=False), nested box {w}x{h}")
                        
                        # Sort the nested bin (will use infinite mode to fill the bin)
                        # Pass start_item_idx so nested bins use correct item indices
                        nested_sorter.sort(nested_start_item_idx=start_item_idx)
                        
                        num_nested_items = len(nested_sorter.bins[0]) if nested_sorter.bins and nested_sorter.bins[0] else 0
                        _debug_print(f"[DEBUG] Nested bin sorted: {num_nested_items} items (scale_factor={scale_factor:.2f}, {exclude_msg})")
                        
                        # If we got at least 2 items, we're done
                        if num_nested_items >= 2:
                            nested_bin_success = True
                            break
                    
                    # If we succeeded, break out of the attempt loop
                    if nested_bin_success:
                        break
                    
                    # If excluding parent ratio didn't work, try allowing it
                    if exclude_parent:
                        exclude_parent = False
                    else:
                        break  # Already tried both approaches
                
                if not nested_bin_success:
                    # Continue to next candidate item
                    continue
                
                if num_nested_items >= 2:
                    # Success! Store the nested bin
                    _debug_print(f"[DEBUG] Nested bin has {num_nested_items} items, storing at (bin_idx={bin_idx}, item_idx={item_idx})")
                    _debug_print(f"[DEBUG] Nested items will start at index {start_item_idx}")
                    
                    # Remap nested item indices to continue from start_item_idx
                    remapped_bin_items = []
                    for nx, ny, nw, nh, nested_item_idx in nested_sorter.bins[0]:
                        # Remap index to continue from start_item_idx
                        new_idx = start_item_idx + nested_item_idx
                        remapped_bin_items.append((nx, ny, nw, nh, new_idx))
                        _debug_print(f"[DEBUG] Remapped nested item {nested_item_idx} -> {new_idx}")
                    nested_sorter.bins = [remapped_bin_items]
                    _debug_print(f"[DEBUG] Before remapping: nested_sorter.nested_bins keys={list(nested_sorter.nested_bins.keys())}")
                    translated_nested_bins = self._remap_nested_bin_data(
                        nested_sorter.nested_bins, start_item_idx, bin_idx, item_idx, (), remapped_bin_items
                    )
                    _debug_print(f"[DEBUG] After remapping: translated_nested_bins keys={list(translated_nested_bins.keys())}")
                    
                    self.nested_bins[(bin_idx, item_idx)] = {
                        'items': remapped_bin_items,
                        'parent_pos': (x, y, w, h),
                        'nested_bins': translated_nested_bins
                    }
                    _debug_print(f"[DEBUG] Nested bin stored successfully. Total nested bins: {len(self.nested_bins)}")
                    nested_bin_created = True
                    break  # Success! Move to next bin
                
                if nested_bin_created:
                    break  # Found a working candidate
            
            if not nested_bin_created:
                _debug_print(f"[DEBUG] WARNING: Could not create nested bin for bin {bin_idx} after all attempts")
    
    def _sort_infinite(self) -> List[List[Tuple]]:
        """Fill a single bin until no item fits or remaining space is below min_space_threshold."""
        if not self.size_ratios:
            return [[]]

        bins = [[]]
        item_idx = 0
        is_first_item = True

        while True:
            largest_rect_area = self._get_largest_fittable_rectangle(bins[-1])
            if largest_rect_area < self.min_space_threshold:
                break

            if is_first_item and (not hasattr(self, '_exclude_ratio_for_first_item') or self._exclude_ratio_for_first_item is None):
                if chance_to_do(self.main_bin_fill_chance):
                    self._exclude_ratio_for_first_item = None
                else:
                    self._exclude_ratio_for_first_item = (self.box_width, self.box_height)

            if is_first_item and getattr(self, '_exclude_ratio_for_first_item', None) is not None:
                _debug_print(f"[DEBUG] [exclude-parent] _sort_infinite: placing first item in bin, exclude_ratio_for_first=True, _exclude_ratio_for_first_item={self._exclude_ratio_for_first_item}, box=({self.box_width},{self.box_height})")
            best = self._find_largest_placeable_item(bins[-1], exclude_ratio_for_first=is_first_item)
            if best is None:
                break

            x, y, w, h, (wr, hr) = best
            area = w * h
            bin_area = self.box_width * self.box_height
            break_threshold = self.item_break_scale * bin_area
            do_break = (
                self.item_break_scale > 0
                and area >= break_threshold
                and chance_to_do(self.item_break_chance)
            )
            if do_break:
                _debug_print(f"[DEBUG] [item-break] Breaking item: pos=({x},{y}) size={w}x{h} area={area} ratio=({wr},{hr}) (threshold={break_threshold:.0f} = {self.item_break_scale*100:.1f}% of bin)")
                bins[-1].append((x, y, w, h, item_idx))
                restricted_ratios = self._get_random_restricted_ratios(wr, hr)
                _debug_print(f"[DEBUG] [item-break] Restricted ratios for nested bin: {[(r[0], r[1]) for r in restricted_ratios]}")
                placements = []
                ratios_only = [(r[0], r[1]) for r in restricted_ratios]
                min_n = max(2, min(self.break_box_min_items, self.break_box_max_items))
                max_n = max(min_n, self.break_box_max_items)
                target = random.randint(min_n, max_n)
                # Try target count first, then target-1, etc. down to min_n; each n gets break_box_fill_attempts tries
                for n in range(target, min_n - 1, -1):
                    for attempt in range(self.break_box_fill_attempts):
                        placements = self._try_layout_n_items(w, h, n, ratios_only)
                        if not placements:
                            nested_sorter = BinSorter(
                                box_size=(w, h),
                                size_ratios=restricted_ratios,
                                min_space_threshold=0,
                                nesting_layers=0,
                                nested_min_space_threshold=self.nested_min_space_threshold,
                                main_bin_fill_chance=0.05,
                                item_break_scale=0,
                                item_break_chance=0.0,
                                break_box_min_items=self.break_box_min_items,
                                break_box_max_items=self.break_box_max_items,
                                break_box_fill_attempts=self.break_box_fill_attempts,
                                break_box_coverage_threshold=self.break_box_coverage_threshold,
                            )
                            nested_sorter._exclude_ratio_for_first_item = (wr, hr)
                            nested_sorter.sort()
                            nested_items = nested_sorter.bins[0] if nested_sorter.bins else []
                            raw = [(nx, ny, nw, nh) for (nx, ny, nw, nh, _) in nested_items]
                            if len(raw) >= n:
                                placements = raw[:n]
                            else:
                                placements = []
                        if placements:
                            coverage = self._coverage_fraction(placements, w, h)
                            if coverage < self.break_box_coverage_threshold:
                                _debug_print(f"[DEBUG] [item-break] Rejected fallback: coverage {coverage:.1%} < {self.break_box_coverage_threshold:.1%}")
                                placements = []
                                continue
                            placements = self._fix_overlaps(placements, w, h)
                            _debug_print(f"[DEBUG] [item-break] Filled with {n} items (attempt {attempt + 1}/{self.break_box_fill_attempts} for n={n})")
                            break
                    if placements:
                        break
                if placements:
                    remapped = [(nx, ny, nw, nh, item_idx + 1 + i) for i, (nx, ny, nw, nh) in enumerate(placements)]
                    _check_overlaps(remapped, "break-box stored items (remapped)")
                    _debug_print(f"[DEBUG] [item-break] Broken into {len(remapped)} items: {[(nx, ny, nw, nh, idx) for (nx, ny, nw, nh, idx) in remapped]}")
                    self.nested_bins[(0, item_idx)] = {
                        'items': remapped,
                        'parent_pos': (x, y, w, h),
                        'nested_bins': {},
                    }
                    item_idx += 1 + len(remapped)
                else:
                    # Break failed (couldn't place min items) - remove the break box item and don't break
                    _debug_print(f"[DEBUG] [item-break] Break failed, removing break box item and not breaking")
                    bins[-1].pop()  # Remove the break box item we just appended
                    bins[-1].append((x, y, w, h, item_idx))  # Add original item instead
                    item_idx += 1
            else:
                bins[-1].append((x, y, w, h, item_idx))
                item_idx += 1
            is_first_item = False

        return bins
    
    def generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        colors = []
        for i in range(num_colors):
            hue = (i * 137.508) % 360  # Golden angle for good distribution
            # Convert HSV to RGB (simplified)
            h = hue / 360.0
            s = 0.7
            v = 0.9
            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c
            
            if h < 1/6:
                r, g, b = c, x, 0
            elif h < 2/6:
                r, g, b = x, c, 0
            elif h < 3/6:
                r, g, b = 0, c, x
            elif h < 4/6:
                r, g, b = 0, x, c
            elif h < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))
        return colors
    
    def _renumber_items_and_log(self) -> None:
        """
        Renumber all item IDs globally to 1..N (main bin first, then nested items) so no two items share an id.
        Update nested_bins keys to use new main-bin IDs. Then print debug info for each item.
        """
        next_id = 1
        for bin_idx, bin_items in enumerate(self.bins):
            if not bin_items:
                continue
            # Main bin: assign IDs 1, 2, ..., n by position
            old_to_new = {bin_items[i][4]: next_id + i for i in range(len(bin_items))}
            self.bins[bin_idx] = [(x, y, w, h, next_id + i) for i, (x, y, w, h, _) in enumerate(bin_items)]
            next_id += len(bin_items)

            # Update nested_bins keys for this bin; assign global IDs to nested items (deterministic order by parent id)
            to_drop = []
            to_add = {}
            for (b_idx, old_idx), data in sorted(self.nested_bins.items(), key=lambda x: (x[0][0], x[0][1])):
                if b_idx != bin_idx:
                    continue
                if old_idx not in old_to_new:
                    continue
                new_idx = old_to_new[old_idx]
                to_drop.append((b_idx, old_idx))
                # Renumber nested items with global IDs (continue from next_id)
                items = data.get('items', [])
                data['items'] = [(nx, ny, nw, nh, next_id + i) for i, (nx, ny, nw, nh, _) in enumerate(items)]
                next_id += len(items)
                to_add[(b_idx, new_idx)] = data
            for k in to_drop:
                del self.nested_bins[k]
            for k, v in to_add.items():
                self.nested_bins[k] = v

        # Debug log: print each item's id, width, height, aspect as fraction, location
        for bin_idx, bin_items in enumerate(self.bins):
            for x, y, w, h, item_id in bin_items:
                g = math.gcd(w, h) if h else 0
                aspect = f"{w // g}/{h // g}" if g else "n/a"
                _debug_print(f"[DEBUG] Item id={item_id} width={w} height={h} aspect={aspect} location=({x},{y})")
            for (b_idx, item_id), nested_data in sorted(self.nested_bins.items()):
                if b_idx != bin_idx:
                    continue
                for nx, ny, nw, nh, nested_id in nested_data.get('items', []):
                    g = math.gcd(nw, nh) if nh else 0
                    aspect = f"{nw // g}/{nh // g}" if g else "n/a"
                    _debug_print(f"[DEBUG] Item id={nested_id} (nested in {item_id}) width={nw} height={nh} aspect={aspect} location=({nx},{ny})")

    def export_image(self, output_path: str = "bin_sorting_result.png", 
                    bins_per_row: int = None, padding: int = 20):
        """
        Export the bin sorting result as a numbered, color-coded image.

        Args:
            output_path: Path to save the image
            bins_per_row: Number of bins per row (auto-calculated if None)
            padding: Padding between bins in pixels
        """
        if not self.bins:
            self.sort()
        self._renumber_items_and_log()
        
        if bins_per_row is None:
            # Use actual number of bins used, not the parameter
            bins_per_row = int(math.ceil(math.sqrt(len(self.bins))))
        
        num_rows = math.ceil(len(self.bins) / bins_per_row)
        # Calculate total image size
        img_width = bins_per_row * (self.box_width + padding) + padding
        img_height = num_rows * (self.box_height + padding) + padding
        
        # Create image
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Generate colors for items
        # Calculate total number of items from bins (works for both modes)
        total_items = sum(len(bin_items) for bin_items in self.bins) if self.bins else 0
        max_items = max(len(bin_items) for bin_items in self.bins) if self.bins else 1
        item_colors = self.generate_colors(max(total_items, 15))
        
        # Global bin counter: starts after main bins, increments for each nested bin
        # Use a list so it can be modified by reference in nested functions
        global_bin_counter = [len(self.bins)]  # Start counting after main bins
        
        # Draw bins
        for bin_idx, bin_items in enumerate(self.bins):
            row = bin_idx // bins_per_row
            col = bin_idx % bins_per_row
            
            offset_x = padding + col * (self.box_width + padding)
            offset_y = padding + row * (self.box_height + padding)
            
            # Draw bin border
            bin_rect = [offset_x, offset_y, 
                       offset_x + self.box_width, offset_y + self.box_height]
            draw.rectangle(bin_rect, outline='black', width=2)
            
            # Draw bin number
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
            except:
                font = ImageFont.load_default()
            
            bin_label = f"Bin {bin_idx + 1}"
            bbox = draw.textbbox((0, 0), bin_label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((offset_x + 5, offset_y - text_height - 5), 
                     bin_label, fill='black', font=font)
            
            # Draw items in bin
            for x, y, w, h, item_idx in bin_items:
                item_x = offset_x + x
                item_y = offset_y + y
                
                # Use color based on item index
                color = item_colors[item_idx % len(item_colors)]
                
                # Draw item rectangle
                item_rect = [item_x, item_y, item_x + w, item_y + h]
                draw.rectangle(item_rect, fill=color, outline='darkblue', width=1)
                
                # Draw nested bin if this item contains one (draw BEFORE item number so it's visible)
                has_nested_bin = (bin_idx, item_idx) in self.nested_bins
                if has_nested_bin:
                    # Pass the global bin counter and initial parent_path for labeling
                    # Unpack the tuple so *parent_path collects it correctly
                    self._draw_nested_bin(draw, item_x, item_y, w, h, 
                                         self.nested_bins[(bin_idx, item_idx)], 
                                         item_colors, font, global_bin_counter, bin_idx, item_idx)
                
                # Draw item number only if it doesn't contain a nested bin (ids are 1-based after _renumber_items_and_log)
                if not has_nested_bin:
                    item_label = str(item_idx)
                    bbox = draw.textbbox((0, 0), item_label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    text_x = item_x + (w - text_w) // 2
                    text_y = item_y + (h - text_h) // 2
                    draw.text((text_x, text_y), item_label, fill='white', font=font)
        
        # Save image
        img.save(output_path)
        print(f"Bin sorting visualization saved to {output_path}")
        return img
    
    def _draw_nested_bin(self, draw, parent_x: int, parent_y: int, parent_w: int, parent_h: int,
                        nested_data: dict, item_colors: List[Tuple], font, global_bin_counter: list, *parent_path):
        """
        Draw a nested bin inside its parent item.
        
        Args:
            global_bin_counter: List with single element [counter] that tracks all bins (main + nested)
            parent_path: Tuple of (bin_idx, item_idx, ...) representing the path to this nested bin
        """
        _debug_print(f"[DEBUG] _draw_nested_bin called with parent_path={parent_path}, parent_pos=({parent_x},{parent_y}), size={parent_w}x{parent_h}")
        nested_items = nested_data['items']
        parent_pos = nested_data['parent_pos']
        nested_bins = nested_data.get('nested_bins', {})
        
        _debug_print(f"[DEBUG] Drawing {len(nested_items)} nested items, {len(nested_bins)} nested bins")
        
        # First pass: Draw all nested items (without their nested bins)
        for nx, ny, nw, nh, nested_item_idx in nested_items:
            nested_item_x = parent_x + nx
            nested_item_y = parent_y + ny
            
            # Use color based on nested item index
            nested_color = item_colors[nested_item_idx % len(item_colors)]
            
            # Draw nested item rectangle
            nested_item_rect = [nested_item_x, nested_item_y, 
                               nested_item_x + nw, nested_item_y + nh]
            draw.rectangle(nested_item_rect, fill=nested_color, outline='darkblue', width=1)
            
            # Draw nested item number (smaller font; ids are 1-based after _renumber_items_and_log)
            nested_label = str(nested_item_idx)
            bbox = draw.textbbox((0, 0), nested_label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = nested_item_x + (nw - text_w) // 2
            text_y = nested_item_y + (nh - text_h) // 2
            if nw > text_w and nh > text_h:  # Only draw if there's space
                draw.text((text_x, text_y), nested_label, fill='white', font=font)
        
        # Second pass: Collect all nested bins (at all depths) and draw them from largest to smallest
        all_nested_bins_to_draw = []  # List of (x, y, w, h, nested_data, nested_key) tuples
        
        def collect_all_nested_bins(nested_bins_dict, base_x, base_y, current_path, items_list):
            """Recursively collect all nested bins with their absolute positions."""
            _debug_print(f"[DEBUG] collect_all_nested_bins: current_path={current_path}, nested_bins_dict keys={list(nested_bins_dict.keys())}, items_list indices={[ni for _, _, _, _, ni in items_list]}")
            for nested_key, nested_data in nested_bins_dict.items():
                # Find the item that contains this nested bin
                # nested_key structure: (..., item_idx) - the last element is the item index
                if len(nested_key) > len(current_path):
                    item_idx = nested_key[-1]
                    _debug_print(f"[DEBUG] Looking for item_idx={item_idx} in items_list")
                    found = False
                    # Find the item in items_list
                    for nx, ny, nw, nh, ni in items_list:
                        if ni == item_idx:
                            _debug_print(f"[DEBUG] Found item {item_idx} at relative_pos=({nx},{ny}), size={nw}x{nh}")
                            abs_x = base_x + nx
                            abs_y = base_y + ny
                            all_nested_bins_to_draw.append((abs_x, abs_y, nw, nh, nested_data, nested_key))
                            found = True
                            
                            # Recursively collect deeper nested bins
                            deeper_nested_bins = nested_data.get('nested_bins', {})
                            if deeper_nested_bins:
                                # Get items from nested_data for deeper level
                                deeper_items = nested_data.get('items', [])
                                collect_all_nested_bins(deeper_nested_bins, abs_x, abs_y, nested_key, deeper_items)
                            break
                    if not found:
                        _debug_print(f"[DEBUG] WARNING: Could not find item_idx={item_idx} in items_list!")
        
        # Collect all nested bins recursively
        collect_all_nested_bins(nested_bins, parent_x, parent_y, parent_path, nested_items)
        
        # Add this level's border to the list (so it's included in size sorting)
        all_nested_bins_to_draw.append((parent_x, parent_y, parent_w, parent_h, nested_data, parent_path))
        
        # Sort all nested bins by size (largest first) so smaller ones draw on top
        all_nested_bins_to_draw.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        # Debug: Print all nested bins that will be drawn
        _debug_print(f"[DEBUG] All nested bins to draw (sorted by size, largest first):")
        for idx, (x, y, w, h, bin_data, bin_key) in enumerate(all_nested_bins_to_draw):
            area = w * h
            print(f"  {idx+1}. Bin key={bin_key}, position=({x},{y}), size={w}x{h}, area={area}")
        
        # Draw all nested bins in order (largest to smallest)
        for x, y, w, h, bin_data, bin_key in all_nested_bins_to_draw:
            # Draw items inside this nested bin first (so they appear behind the border).
            # Skip items for the current level (bin_key == parent_path); those were already drawn in the first pass.
            if bin_key != parent_path:
                for nx, ny, nw, nh, item_idx in bin_data.get('items', []):
                    item_x = x + nx
                    item_y = y + ny
                    color = item_colors[item_idx % len(item_colors)]
                    draw.rectangle([item_x, item_y, item_x + nw, item_y + nh],
                                  fill=color, outline='darkblue', width=1)
                    item_label = str(item_idx)
                    bbox = draw.textbbox((0, 0), item_label, font=font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    text_x = item_x + (nw - text_w) // 2
                    text_y = item_y + (nh - text_h) // 2
                    if nw > text_w and nh > text_h:
                        draw.text((text_x, text_y), item_label, fill='white', font=font)
            # Draw the border and label for this nested bin
            nested_bin_rect = [x, y, x + w, y + h]
            draw.rectangle(nested_bin_rect, outline=(0, 255, 255), width=3)  # Cyan border
            
            # Draw nested bin number
            global_bin_counter[0] += 1
            nested_bin_number = global_bin_counter[0]
            nested_bin_label = f"Bin {nested_bin_number}"
            bbox = draw.textbbox((0, 0), nested_bin_label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x + 5, y - text_height - 5), 
                     nested_bin_label, fill=(0, 255, 255), font=font)
            _debug_print(f"[DEBUG] Drew nested bin '{nested_bin_label}' at ({x}, {y}), size={w}x{h}, bin_key={bin_key}")


def main():
    """Example usage of the bin sorter."""
    # Customizable parameters
    BOX_SIZE = ((1920),(1920))  # pixels (width, height)
    # Width:height:weight ratios; optional (wr, hr, weight, expand_x, expand_y) for gap-fill (e.g. 0.1 = 10% expand)
    SIZE_RATIOS = [
        (1,1, 0.3, .2, .2),
        (1,2, 0.1, .2, .2),
        (2,3, 0.2, .2, .2),
        (1,3, 0.05, .2, .2),
        (2,1, 0.1, .2, .2),
        (3,2, 0.2, .2, .2),
        (3,1, 0.05, .2, .2)
    ]

    VERBOSE = True
    global _VERBOSE
    _VERBOSE = VERBOSE
    if VERBOSE:
        print(f"Verbose mode: ON")
    else:
        print(f"Verbose mode: OFF")
    
    MIN_SPACE_THRESHOLD = 0  # Minimum space area (in pixels) to consider bin full
    
    # Nesting configuration
    NESTING_LAYERS = 1 # Number of nesting layers (0 = no nesting)
    NESTED_MIN_SPACE_THRESHOLD = 0  # Minimum space threshold for nested bins
    
    # Main bin first-item: chance (0–1) to allow ratio that fills the box; else that ratio is excluded
    MAIN_BIN_FILL_CHANCE = 0.05  # 5% allow one-item-fill, 95% exclude it

    # Item break: large items can be turned into nested bins of smaller items
    ITEM_BREAK_SCALE = 0.45  # Fraction (0–1) of bin area; e.g. 0.25 = break when item >= 25% of bin. 0 = disabled
    ITEM_BREAK_CHANCE = 0.95  # Probability (0–1) to break when item area >= ITEM_BREAK_SCALE * bin_area
    BREAK_BOX_MIN_ITEMS = 1  # Minimum number of items to place in a break box
    BREAK_BOX_MAX_ITEMS = 4  # Maximum number of items to place in a break box
    BREAK_BOX_FILL_ATTEMPTS = 5  # Retry fill this many times; accept first that yields valid placements
    BREAK_BOX_COVERAGE_THRESHOLD = 0.99  # Minimum coverage for fallback layouts; reject if < threshold

    # Output folder for exported images (use "." for current directory)
    OUTPUT_PATH = "../taking_stock_production/bin_sorting"
    if not os.path.isdir(OUTPUT_PATH):
        try:
            os.makedirs(OUTPUT_PATH, exist_ok=True)
        except OSError as e:
            print(f"Error: could not create output folder {OUTPUT_PATH}: {e}", file=sys.stderr)
            sys.exit(1)

    # Batch exports: run and export this many times
    BATCH_AMOUNT = 50
    box_w, box_h = BOX_SIZE[0], BOX_SIZE[1]
    g = math.gcd(box_w, box_h)
    aspect = f"{box_w // g}_{box_h // g}" if g else "1_1"

    for batch_idx in range(BATCH_AMOUNT):
        # Create and run bin sorter
        sorter = BinSorter(BOX_SIZE, SIZE_RATIOS,
                          min_space_threshold=MIN_SPACE_THRESHOLD,
                          nesting_layers=NESTING_LAYERS,
                          nested_min_space_threshold=NESTED_MIN_SPACE_THRESHOLD,
                          main_bin_fill_chance=MAIN_BIN_FILL_CHANCE,
                          item_break_scale=ITEM_BREAK_SCALE,
                          item_break_chance=ITEM_BREAK_CHANCE,
                          break_box_min_items=BREAK_BOX_MIN_ITEMS,
                          break_box_max_items=BREAK_BOX_MAX_ITEMS,
                          break_box_fill_attempts=BREAK_BOX_FILL_ATTEMPTS,
                          break_box_coverage_threshold=BREAK_BOX_COVERAGE_THRESHOLD)
        sorter.sort()

        if BATCH_AMOUNT <= 1:
            file_name = f"bin_nest_{NESTING_LAYERS}_aspect_{aspect}.png"
        else:
            file_name = f"bin_nest_{NESTING_LAYERS}_aspect_{aspect}_{batch_idx + 1:04d}.png"
        full_path = os.path.join(OUTPUT_PATH, file_name)
        sorter.export_image(full_path, bins_per_row=1, padding=100)

        total_items = sorter.get_total_items_count()
        main_items = sum(len(bin_items) for bin_items in sorter.bins)
        nested_items = total_items - main_items
        total_bins = sorter.get_total_bins_count()
        main_bins = len(sorter.bins)
        nested_bins_count = total_bins - main_bins

        if BATCH_AMOUNT > 1:
            print(f"  [{batch_idx + 1}/{BATCH_AMOUNT}] {file_name}  bins={total_bins}  items={total_items}")
        else:
            print(f"\nBin Sorting Complete!")
            print(f"Nesting layers: {NESTING_LAYERS}")
            print(f"file saved as: {full_path}")
            print(f"Number of bins: {total_bins} (main: {main_bins}, nested: {nested_bins_count})")
            print(f"Box size: {BOX_SIZE[0]}x{BOX_SIZE[1]} pixels")
            print(f"Minimum space threshold: {MIN_SPACE_THRESHOLD} pixels")
            print(f"Total items placed: {total_items} (main: {main_items}, nested: {nested_items})")
            print(f"Items placed in {total_bins} bins")

    if BATCH_AMOUNT > 1:
        print(f"\nBatch complete: {BATCH_AMOUNT} exports saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
