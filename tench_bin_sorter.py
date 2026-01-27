import os
import sys
import random
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import math


class BinSorter:
    def __init__(self, num_bins: int, box_size: Tuple[int, int], size_ratios: List[Tuple[int, int, float]], 
                 num_items: int, infinite_items: bool = False, min_space_threshold: int = 100,
                 nesting_layers: int = 0, nested_min_space_threshold: int = 100,
                 main_bin_fill_chance: float = 0.05):
        """
        Initialize the bin sorter.
        
        Args:
            num_bins: Number of bins to create
            box_size: Size of the container box in pixels (width, height)
            size_ratios: List of (width, height, weight) ratios for items to pack, e.g., [(2, 3, 1.0), (1, 1, 1.0), (4, 3, 0.5)]
                         Weight determines how likely the ratio is to be selected (higher = more likely)
            num_items: Number of items to create (ignored if infinite_items=True)
            infinite_items: If True, keep placing items until bin is full (based on min_space_threshold)
            min_space_threshold: Minimum space area required to place new items (used in infinite mode)
            nesting_layers: Number of nesting layers to create (0 = no nesting)
            nested_min_space_threshold: Minimum space threshold for nested bins
            main_bin_fill_chance: Probability (0–1) that the main bin's first item may use the ratio that fills
                                 the entire box. E.g. 0.05 = 5% allow full-bin ratio, 95% exclude it.
        """
        self.num_bins = num_bins
        self.box_width, self.box_height = box_size
        self.size_ratios = size_ratios
        self.num_items = num_items
        self.infinite_items = infinite_items
        self.min_space_threshold = min_space_threshold
        self.nesting_layers = nesting_layers
        self.nested_min_space_threshold = nested_min_space_threshold
        self.main_bin_fill_chance = main_bin_fill_chance
        self.bins = []
        self.nested_bins = {}  # Maps (bin_idx, item_idx) -> nested bin data
        self._retry_scale_factor = 1.0  # Scale factor for retries (reduced on each retry)
        
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
        
    def create_items(self) -> List[Tuple[float, float, int]]:
        """
        Create items from width:height ratios.
        Randomly selects from size_ratios to create num_items items.
        Iteratively scales items down until they all fit in one bin.
        Returns list of (width, height, index) tuples.
        """
        if not self.size_ratios:
            return []
            
        # Calculate single bin area (scale to fill one bin, not all bins)
        single_bin_area = self.box_width * self.box_height
        
        # First pass: randomly select ratios and calculate ratio areas for all items
        ratio_areas = []
        max_width_ratio = 0
        max_height_ratio = 0
        selected_ratios = []
        
        for idx in range(self.num_items):
            width_ratio, height_ratio = self._select_weighted_ratio()
            selected_ratios.append((width_ratio, height_ratio))
            ratio_area = width_ratio * height_ratio
            ratio_areas.append(ratio_area)
            max_width_ratio = max(max_width_ratio, width_ratio)
            max_height_ratio = max(max_height_ratio, height_ratio)
        
        # Calculate total ratio area
        total_ratio_area = sum(ratio_areas)
        
        if total_ratio_area == 0:
            return []
        
        # Start with scale that fills one bin's area
        scale_by_area = math.sqrt(single_bin_area / total_ratio_area) * self._retry_scale_factor
        
        # Also calculate scale to ensure no single item exceeds bin dimensions
        scale_by_dimension = min(
            self.box_width / max_width_ratio if max_width_ratio > 0 else float('inf'),
            self.box_height / max_height_ratio if max_height_ratio > 0 else float('inf')
        )
        
        # Start with the smaller scale
        scale = min(scale_by_area, scale_by_dimension)
        
        # Iteratively scale down until all items fit in one bin
        max_iterations = 50
        scale_reduction = 0.95  # Reduce scale by 5% each iteration
        
        for iteration in range(max_iterations):
            # Create items with current scale
            items = []
            for idx in range(self.num_items):
                width_ratio, height_ratio = selected_ratios[idx]
                
                width = int(width_ratio * scale)
                height = int(height_ratio * scale)
                
                # Ensure minimum size of 1 pixel
                if width < 1:
                    width = 1
                if height < 1:
                    height = 1
                    
                items.append((width, height, idx))
            
            # Try packing items with current scale
            # Temporarily set num_bins to 1 to test if they fit
            original_num_bins = self.num_bins
            self.num_bins = 1
            test_bins = self.first_fit_decreasing(items)
            self.num_bins = original_num_bins
            
            # If all items fit in one bin, we're done
            if len(test_bins) <= 1:
                return items
            
            # Otherwise, scale down and try again
            scale *= scale_reduction
        
        # If we couldn't fit in one bin after max iterations, return items with final scale
        return items
    
    def first_fit_decreasing(self, items: List[Tuple[float, float, int]]) -> List[List[Tuple]]:
        """
        First Fit Decreasing bin packing algorithm.
        Sorts items by area (largest first) and places them in the first bin that fits.
        
        Returns:
            List of bins, where each bin is a list of (x, y, width, height, item_idx) tuples
        """
        # Sort items by area (decreasing)
        sorted_items = sorted(items, key=lambda x: x[0] * x[1], reverse=True)
        
        bins = []
        
        for item_width, item_height, item_idx in sorted_items:
            placed = False
            
            # Try to place in existing bins
            for bin_idx, bin_items in enumerate(bins):
                if self._can_place_in_bin(bin_items, item_width, item_height, 
                                         self.box_width, self.box_height):
                    x, y = self._find_position(bin_items, item_width, item_height,
                                             self.box_width, self.box_height)
                    if x is not None:
                        bin_items.append((x, y, item_width, item_height, item_idx))
                        placed = True
                        break
            
            # If couldn't place, create new bin
            if not placed:
                if len(bins) < self.num_bins:
                    bins.append([(0, 0, item_width, item_height, item_idx)])
                else:
                    # If we've exceeded num_bins, still try to place (overflow)
                    bins.append([(0, 0, item_width, item_height, item_idx)])
        
        return bins
    
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
                print(f"[DEBUG] Excluding parent ratio {exclude_ratio} for first nested item")

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
                        print(f"[DEBUG] Skipping excluded ratio ({width_ratio}, {height_ratio}) [box-fill aspect]")
                        continue
                elif (width_ratio == parent_w and height_ratio == parent_h):
                    print(f"[DEBUG] Skipping excluded ratio ({width_ratio}, {height_ratio})")
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
                print(f"[DEBUG] Selected ratio {best_candidate[4]} with area={best_candidate[6]}, weight={best_candidate[7]}, score={best_candidate[5]:.0f}")

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
        """Run the bin sorting algorithm.
        
        Args:
            nested_start_item_idx: Starting item index for nested bins (used when this is a nested sorter)
        """
        print(f"\n[DEBUG] sort() called: nesting_layers={self.nesting_layers}, infinite_items={self.infinite_items}")
        
        max_retries = 10  # Prevent infinite loops
        retry_count = 0
        
        while retry_count < max_retries:
            if self.infinite_items:
                self.bins = self._sort_infinite()
            else:
                items = self.create_items()
                self.bins = self.first_fit_decreasing(items)
            
            if self.nesting_layers > 0:
                print(f"[DEBUG] Initial sorting complete: {len(self.bins)} bins created")
                for i, bin_items in enumerate(self.bins):
                    print(f"  Bin {i}: {len(bin_items)} items")
            
            # Check if main bin has only 1 item (only check first bin, and only in non-infinite mode)
            if not self.infinite_items and self.bins and len(self.bins) == 1:
                total_items = len(self.bins[0])
                if total_items == 1:
                    # 95% chance to retry with scaled down items
                    if random.random() < 0.95:
                        retry_count += 1
                        # Reduce scale factor to make items smaller (allows more items to fit)
                        self._retry_scale_factor *= 0.7  # Reduce scale by 30% each retry
                        continue
            
            # If we get here, either we have multiple items or we're not retrying
            break
        
        # Create nested bins if nesting_layers > 0
        if self.nesting_layers > 0:
            print(f"[DEBUG] Creating nested bins with {self.nesting_layers} layers")
            # Use provided start_item_idx if this is a nested sorter, otherwise calculate it
            if nested_start_item_idx is not None:
                self._create_nested_bins(self.nesting_layers, nested_start_item_idx)
            else:
                self._create_nested_bins(self.nesting_layers)
        
        print(f"[DEBUG] Final nested_bins count: {len(self.nested_bins)}")
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
        # Count main bins
        main_bins = len(self.bins)
        
        # Count nested bins (recursively)
        nested_bins_count = self._count_nested_bins(self.nested_bins)
        
        return main_bins + nested_bins_count
    
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
                print(f"[DEBUG] _remap_nested_bin_data: old_item_idx={old_item_idx}, remapped_items_list indices={[ni for _, _, _, _, ni in remapped_items_list]}")
                # old_item_idx is already a remapped index from nested_sorter.bins[0]
                # which matches the indices in remapped_items_list, so use it directly
                remapped_item_idx = old_item_idx
                print(f"[DEBUG] _remap_nested_bin_data: using old_item_idx={old_item_idx} directly as remapped_item_idx")
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
            
            # Recursively remap deeper nested bins
            if 'nested_bins' in nested_data and nested_data['nested_bins']:
                remapped_data['nested_bins'] = self._remap_nested_bin_data(
                    nested_data['nested_bins'], offset, bin_idx, parent_item_idx, remapped_key, remapped_items_list
                )
            
            remapped[remapped_key] = remapped_data
        
        return remapped
    
    def _create_nested_bins(self, remaining_layers: int, start_item_idx: int = None):
        """Recursively create nested bins inside items. Always succeeds by trying different approaches.
        
        Args:
            remaining_layers: Number of nesting layers remaining
            start_item_idx: Starting item index for nested items (None = calculate from main bins)
        """
        print(f"\n[DEBUG] _create_nested_bins called with remaining_layers={remaining_layers}, start_item_idx={start_item_idx}")
        print(f"[DEBUG] Total bins: {len(self.bins)}")
        
        # Calculate starting index if not provided
        if start_item_idx is None:
            start_item_idx = sum(len(bin_items) for bin_items in self.bins)
            print(f"[DEBUG] Calculated start_item_idx: {start_item_idx}")
        
        if remaining_layers <= 0:
            print(f"[DEBUG] No more nesting layers remaining, returning")
            return
        
        # Process each bin
        for bin_idx, bin_items in enumerate(self.bins):
            print(f"\n[DEBUG] Processing bin {bin_idx} with {len(bin_items)} items")
            
            if not bin_items:
                print(f"[DEBUG] Bin {bin_idx} is empty, skipping")
                continue
            
            # Find the largest items (up to 2)
            items_with_area = [(item, item[2] * item[3]) for item in bin_items]
            items_with_area.sort(key=lambda x: x[1], reverse=True)
            
            print(f"[DEBUG] Top items by area:")
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
                print(f"[DEBUG] Trying item {item_idx}: size={w}x{h}, all bin item indices: {[ni for _, _, _, _, ni in bin_items]}")
                
                # Determine parent item's ratio (exact match only, no rotation)
                parent_ratio = self._get_item_ratio(w, h)
                if parent_ratio:
                    parent_w_ratio, parent_h_ratio = parent_ratio
                    print(f"[DEBUG] Parent item ratio: ({parent_w_ratio}, {parent_h_ratio}) - will exclude for first nested item")
                else:
                    parent_w_ratio, parent_h_ratio = None, None
                
                # Create nested sorter with infinite_items=True
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
                            num_bins=1,
                            box_size=(w, h),
                            size_ratios=self.size_ratios,
                            num_items=self.num_items,  # This will be ignored in infinite mode
                            infinite_items=True,
                            min_space_threshold=scaled_threshold,
                            nesting_layers=remaining_layers - 1,
                            nested_min_space_threshold=self.nested_min_space_threshold
                        )
                        
                        # Store parent ratio in nested sorter so it can exclude it for first item
                        if exclude_parent and parent_w_ratio is not None and parent_h_ratio is not None:
                            nested_sorter._exclude_ratio_for_first_item = (parent_w_ratio, parent_h_ratio)
                            exclude_msg = "excluding parent ratio"
                        else:
                            nested_sorter._exclude_ratio_for_first_item = None
                            exclude_msg = "allowing parent ratio"
                        
                        # Sort the nested bin (will use infinite mode to fill the bin)
                        # Pass start_item_idx so nested bins use correct item indices
                        nested_sorter.sort(nested_start_item_idx=start_item_idx)
                        
                        num_nested_items = len(nested_sorter.bins[0]) if nested_sorter.bins and nested_sorter.bins[0] else 0
                        print(f"[DEBUG] Nested bin sorted: {num_nested_items} items (scale_factor={scale_factor:.2f}, {exclude_msg})")
                        
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
                    print(f"[DEBUG] Nested bin has {num_nested_items} items, storing at (bin_idx={bin_idx}, item_idx={item_idx})")
                    print(f"[DEBUG] Nested items will start at index {start_item_idx}")
                    
                    # Remap nested item indices to continue from start_item_idx
                    remapped_bin_items = []
                    for nx, ny, nw, nh, nested_item_idx in nested_sorter.bins[0]:
                        # Remap index to continue from start_item_idx
                        new_idx = start_item_idx + nested_item_idx
                        remapped_bin_items.append((nx, ny, nw, nh, new_idx))
                        print(f"[DEBUG] Remapped nested item {nested_item_idx} -> {new_idx}")
                    
                    # Update nested_sorter with remapped items for recursive nesting
                    nested_sorter.bins = [remapped_bin_items]
                    
                    # The nested_sorter.sort() already created nesting for nesting_layers-1 levels
                    # because we set nesting_layers=remaining_layers-1 when creating the nested_sorter.
                    # However, if remaining_layers > 1, we need to ensure deeper nesting happens.
                    # The nested_sorter's sort() should have already created nested bins, but
                    # those nested bins use the original item indices. We need to:
                    # 1. Remap the nested bin keys to use the new item indices
                    # 2. Ensure deeper nesting is created if remaining_layers > 1
                    
                    # If remaining_layers > 1, ensure deeper nesting happens
                    # The nested_sorter should have already created nesting for nesting_layers levels,
                    # but we need to make sure deeper nesting happens inside those nested bins.
                    # Calculate next starting index (after current nested items)
                    next_start_idx = start_item_idx + len(remapped_bin_items)
                    if remaining_layers > 1:
                        print(f"[DEBUG] Ensuring deeper nesting: remaining_layers={remaining_layers}, will create {remaining_layers - 1} more levels")
                        print(f"[DEBUG] Nested sorter currently has {len(nested_sorter.nested_bins)} nested bins")
                        print(f"[DEBUG] nested_sorter.bins[0] item indices before recursive call: {[ni for _, _, _, _, ni in nested_sorter.bins[0]]}")
                        # Create deeper nesting inside the nested items
                        # This will process the remapped items and create nested bins inside them
                        nested_sorter._create_nested_bins(remaining_layers - 1, next_start_idx)
                        print(f"[DEBUG] nested_sorter.bins[0] item indices after recursive call: {[ni for _, _, _, _, ni in nested_sorter.bins[0]]}")
                        print(f"[DEBUG] nested_sorter.nested_bins keys after recursive call: {list(nested_sorter.nested_bins.keys())}")
                    
                    # Translate nested bin keys with remapped indices using recursive helper
                    # This handles the remapping for the parent's nested_bins dictionary
                    # Pass the original nested_sorter.nested_bins (before remapping) so _remap_nested_bin_data
                    # can properly remap all indices including deeper nested bins
                    # Also pass remapped_bin_items so it can map old_item_idx to correct remapped index
                    print(f"[DEBUG] Before remapping: nested_sorter.nested_bins keys={list(nested_sorter.nested_bins.keys())}")
                    translated_nested_bins = self._remap_nested_bin_data(
                        nested_sorter.nested_bins, start_item_idx, bin_idx, item_idx, (), remapped_bin_items
                    )
                    print(f"[DEBUG] After remapping: translated_nested_bins keys={list(translated_nested_bins.keys())}")
                    
                    self.nested_bins[(bin_idx, item_idx)] = {
                        'items': remapped_bin_items,
                        'parent_pos': (x, y, w, h),
                        'nested_bins': translated_nested_bins
                    }
                    print(f"[DEBUG] Nested bin stored successfully. Total nested bins: {len(self.nested_bins)}")
                    nested_bin_created = True
                    break  # Success! Move to next bin
                
                if nested_bin_created:
                    break  # Found a working candidate
            
            if not nested_bin_created:
                print(f"[DEBUG] WARNING: Could not create nested bin for bin {bin_idx} after all attempts")
    
    def _sort_infinite(self) -> List[List[Tuple]]:
        """Sort items in infinite mode - keep placing until bin is full."""
        if not self.size_ratios:
            return []

        # Start with one empty bin
        bins = [[]]
        item_idx = 0
        is_first_item = True  # Track if this is the first item in the bin

        while len(bins) <= self.num_bins:
            # Check if current bin has enough space before trying to place
            if bins:
                largest_rect_area = self._get_largest_fittable_rectangle(bins[-1])
                if largest_rect_area < self.min_space_threshold:
                    # Current bin is full enough
                    if len(bins) < self.num_bins:
                        bins.append([])
                        is_first_item = True  # Reset for new bin
                        continue
                    else:
                        # No more bins, we're done
                        break

            # For first item in main bin: main_bin_fill_chance to allow ratio that fills the box, else exclude it
            if is_first_item and (not hasattr(self, '_exclude_ratio_for_first_item') or self._exclude_ratio_for_first_item is None):
                if random.random() < self.main_bin_fill_chance:
                    self._exclude_ratio_for_first_item = None  # allow full-bin ratio
                else:
                    self._exclude_ratio_for_first_item = (self.box_width, self.box_height)  # exclude box-fill ratio
            
            # Place the largest possible item next (greedy by area).
            best = self._find_largest_placeable_item(bins[-1] if bins else [], 
                                                    exclude_ratio_for_first=is_first_item)
            if best is None:
                # Nothing can fit; treat as full-enough and advance/stop.
                if len(bins) < self.num_bins:
                    bins.append([])
                    is_first_item = True  # Reset for new bin
                    continue
                break

            x, y, w, h, _ratio = best
            bins[-1].append((x, y, w, h, item_idx))
            item_idx += 1
            is_first_item = False  # After first item, allow all ratios
        
        self.bins = bins
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
                
                # Draw item number only if it doesn't contain a nested bin
                if not has_nested_bin:
                    item_label = str(item_idx + 1)
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
        print(f"[DEBUG] _draw_nested_bin called with parent_path={parent_path}, parent_pos=({parent_x},{parent_y}), size={parent_w}x{parent_h}")
        nested_items = nested_data['items']
        parent_pos = nested_data['parent_pos']
        nested_bins = nested_data.get('nested_bins', {})
        
        print(f"[DEBUG] Drawing {len(nested_items)} nested items, {len(nested_bins)} nested bins")
        
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
            
            # Draw nested item number (smaller font)
            nested_label = str(nested_item_idx + 1)
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
            print(f"[DEBUG] collect_all_nested_bins: current_path={current_path}, nested_bins_dict keys={list(nested_bins_dict.keys())}, items_list indices={[ni for _, _, _, _, ni in items_list]}")
            for nested_key, nested_data in nested_bins_dict.items():
                # Find the item that contains this nested bin
                # nested_key structure: (..., item_idx) - the last element is the item index
                if len(nested_key) > len(current_path):
                    item_idx = nested_key[-1]
                    print(f"[DEBUG] Looking for item_idx={item_idx} in items_list")
                    found = False
                    # Find the item in items_list
                    for nx, ny, nw, nh, ni in items_list:
                        if ni == item_idx:
                            print(f"[DEBUG] Found item {item_idx} at relative_pos=({nx},{ny}), size={nw}x{nh}")
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
                        print(f"[DEBUG] WARNING: Could not find item_idx={item_idx} in items_list!")
        
        # Collect all nested bins recursively
        collect_all_nested_bins(nested_bins, parent_x, parent_y, parent_path, nested_items)
        
        # Add this level's border to the list (so it's included in size sorting)
        all_nested_bins_to_draw.append((parent_x, parent_y, parent_w, parent_h, nested_data, parent_path))
        
        # Sort all nested bins by size (largest first) so smaller ones draw on top
        all_nested_bins_to_draw.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        # Debug: Print all nested bins that will be drawn
        print(f"[DEBUG] All nested bins to draw (sorted by size, largest first):")
        for idx, (x, y, w, h, bin_data, bin_key) in enumerate(all_nested_bins_to_draw):
            area = w * h
            print(f"  {idx+1}. Bin key={bin_key}, position=({x},{y}), size={w}x{h}, area={area}")
        
        # Draw all nested bins in order (largest to smallest)
        for x, y, w, h, bin_data, bin_key in all_nested_bins_to_draw:
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
            print(f"[DEBUG] Drew nested bin '{nested_bin_label}' at ({x}, {y}), size={w}x{h}, bin_key={bin_key}")


def main():
    """Example usage of the bin sorter."""
    # Customizable parameters
    NUM_BINS = 1
    BOX_SIZE = ((1920*4), (1920))  # pixels (width, height)
    NUM_ITEMS = 7  # Number of items to create (ignored if INFINITE_ITEMS=True)
    # Width:height:weight ratios for items, e.g., (2, 3, 1.0) means width:height = 2:3 with weight 1.0
    # Weight determines selection probability (higher = more likely to be picked)
    # Default: equal weights (1.0 each) - customize individual weights as needed
    SIZE_RATIOS = [
        (1,1, 0.3),
        (1,2, 0.1),
        (2,3, 0.2),
        (1,3, 0.05),
        (2,1, 0.1),
        (3,2, 0.2),
        (3,1, 0.05)
    ]


    
    # Infinite items mode
    INFINITE_ITEMS = True  # Set to True to fill bin completely
    MIN_SPACE_THRESHOLD = 30000  # Minimum space area (in pixels) to consider bin full (used when INFINITE_ITEMS=True)
    
    # Nesting configuration
    NESTING_LAYERS = 1  # Number of nesting layers (0 = no nesting)
    NESTED_MIN_SPACE_THRESHOLD = 1000  # Minimum space threshold for nested bins
    
    # Main bin first-item: chance (0–1) to allow ratio that fills the box; else that ratio is excluded
    MAIN_BIN_FILL_CHANCE = 0.05  # 5% allow one-item-fill, 95% exclude it
    
    # Output folder for exported images (use "." for current directory)
    OUTPUT_PATH = "../taking_stock_production/bin_sorting"
    if not os.path.isdir(OUTPUT_PATH):
        try:
            os.makedirs(OUTPUT_PATH, exist_ok=True)
        except OSError as e:
            print(f"Error: could not create output folder {OUTPUT_PATH}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Create and run bin sorter
    sorter = BinSorter(NUM_BINS, BOX_SIZE, SIZE_RATIOS, NUM_ITEMS, 
                      infinite_items=INFINITE_ITEMS, 
                      min_space_threshold=MIN_SPACE_THRESHOLD,
                      nesting_layers=NESTING_LAYERS,
                      nested_min_space_threshold=NESTED_MIN_SPACE_THRESHOLD,
                      main_bin_fill_chance=MAIN_BIN_FILL_CHANCE)
    sorter.sort()
    
    # Export image to OUTPUT_PATH
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    file_name = f"bin_sorting_result_nested_{NESTING_LAYERS}_{MIN_SPACE_THRESHOLD}_{BOX_SIZE[0]}x{BOX_SIZE[1]}.png"
    full_path = os.path.join(OUTPUT_PATH, file_name)
    sorter.export_image(full_path, bins_per_row=1, padding=100)
    
    # Count total items placed (including nested items)
    total_items = sorter.get_total_items_count()
    main_items = sum(len(bin_items) for bin_items in sorter.bins)
    nested_items = total_items - main_items
    
    # Count total bins (including nested bins)
    total_bins = sorter.get_total_bins_count()
    main_bins = len(sorter.bins)
    nested_bins_count = total_bins - main_bins
    
    print(f"\nBin Sorting Complete!")
    print(f"file saved as: {full_path}")
    print(f"Number of bins: {total_bins} (main: {main_bins}, nested: {nested_bins_count})")
    print(f"Box size: {BOX_SIZE[0]}x{BOX_SIZE[1]} pixels")
    print(f"Infinite items mode: {INFINITE_ITEMS}")
    if not INFINITE_ITEMS:
        print(f"Number of items requested: {NUM_ITEMS}")
    else:
        print(f"Minimum space threshold: {MIN_SPACE_THRESHOLD} pixels")
    print(f"Total items placed: {total_items} (main: {main_items}, nested: {nested_items})")
    print(f"Items placed in {total_bins} bins")


if __name__ == "__main__":
    main()
