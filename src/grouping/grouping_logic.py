import os
import sys
import json
import glob
import time
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, deque, Counter

try:
    # This assumes grouping_logic.py is in 'project_root/src/grouping/'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from configs.config_manager import get_config, get_config_value
except ImportError:
    print("Warning: Could not import config_manager. Using default values.")

    def get_config(name):
        return {}

class BoundingBoxGrouper:
    def __init__(self):
        try:
            self.grouping_config = get_config('grouping')
            self.high_overlap_threshold = self.grouping_config.get("high_overlap_threshold", 0.1)
            self.high_iou_threshold = self.grouping_config.get("high_iou_threshold", 0.95)
            self.contained_box_threshold = self.grouping_config.get("contained_box_threshold", 0.95)
            self.vertical_alignment_factor = self.grouping_config.get("vertical_alignment_factor", 0.2)
            self.horizontal_alignment_factor = self.grouping_config.get("horizontal_alignment_factor", 0.2)
            self.proximity_factor = self.grouping_config.get("proximity_factor", 1)
            self.neighborhood_expansion_factor = self.grouping_config.get("neighborhood_expansion_factor", 1)

            data_loader_config = get_config('data_loader')
            detection_dir_relative = data_loader_config.get("detection_metadata_dir",
                                                            "data/processed/metadata/detection_metadata")
            group_dir_relative = data_loader_config.get("group_detection_metadata_dir",
                                                        "data/processed/metadata/group_detection_metadata")
        except Exception as e:
            print(f"Warning: Error loading config: {e}. Using default values.")
            # Default values if config is not available
            self.high_overlap_threshold = 0.1
            self.high_iou_threshold = 0.95
            self.contained_box_threshold = 0.95
            self.vertical_alignment_factor = 0.2
            self.horizontal_alignment_factor = 0.2
            self.proximity_factor = 1
            self.neighborhood_expansion_factor = 1
            detection_dir_relative = "data/processed/metadata/detection_metadata"
            group_dir_relative = "data/processed/metadata/group_detection_metadata"

        # Get project root for fallback case
        try:
            project_root_fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        except:
            project_root_fallback = os.getcwd()

        self.detection_dir = os.path.join(project_root_fallback, detection_dir_relative)
        self.group_dir = os.path.join(project_root_fallback, group_dir_relative)

        # Create output directory if it doesn't exist
        os.makedirs(self.group_dir, exist_ok=True)

        # Initialize metrics tracking
        self.metrics = {
            'total_images_processed': 0,
            'total_boxes_before_grouping': 0,
            'total_boxes_after_grouping': 0,
            'total_merges_performed': 0,
            'processing_times': []
        }

    def load_json_files(self, folder_path: str) -> List[Dict[str, Any]]:
        """Load all *_ocr.json files from a folder"""
        json_files = glob.glob(os.path.join(folder_path, "*_ocr.json"))
        all_data = []

        print(f"    Found {len(json_files)} JSON files in {folder_path}")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                        print(f"    Loaded {len(data)} items from {os.path.basename(json_file)}")
                    else:
                        all_data.append(data)
                        print(f"    Loaded 1 item from {os.path.basename(json_file)}")
            except Exception as e:
                print(f"    Error loading {json_file}: {e}")

        return all_data

    def convert_to_global_coordinates(self, bbox: List[List[int]], tile_coordinates: List[int]) -> List[List[int]]:
        """Convert local bbox coordinates to global coordinates using tile_coordinates"""
        if len(tile_coordinates) < 2:
            print(f"    Warning: Invalid tile_coordinates format: {tile_coordinates}")
            return bbox  # Return original bbox if conversion fails

        try:
            tile_x_min, tile_y_min = tile_coordinates[0], tile_coordinates[1]
            global_bbox = []

            for point in bbox:
                if len(point) >= 2:
                    global_x = point[0] + tile_x_min
                    global_y = point[1] + tile_y_min
                    global_bbox.append([global_x, global_y])
                else:
                    print(f"    Warning: Invalid bbox point format: {point}")
                    global_bbox.append(point)

            return global_bbox
        except Exception as e:
            print(f"    Error in coordinate conversion: {e}")
            return bbox  # Return original bbox if conversion fails

    def calculate_box_parameters(self, bbox: List[List[int]]) -> Dict[str, Any]:
        """Calculate x_min, x_max, y_min, y_max, center for a bounding box"""
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'center_x': center_x, 'center_y': center_y,
            'width': width, 'height': height
        }

    def calculate_iou(self, box1_params: Dict, box2_params: Dict) -> float:
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Calculate intersection area
        x_left = max(box1_params['x_min'], box2_params['x_min'])
        y_top = max(box1_params['y_min'], box2_params['y_min'])
        x_right = min(box1_params['x_max'], box2_params['x_max'])
        y_bottom = min(box1_params['y_max'], box2_params['y_max'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = box1_params['width'] * box1_params['height']
        box2_area = box2_params['width'] * box2_params['height']
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def calculate_intersection_over_little(self, box1_params: Dict, box2_params: Dict) -> Tuple[float, float]:
        """
        Calculate Intersection-over-Little (IoL) for both boxes.
        IoL(A_in_B) = Area(Intersection) / Area(A) - How much of A is in B?
        IoL(B_in_A) = Area(Intersection) / Area(B) - How much of B is in A?
        Returns (IoL_1_in_2, IoL_2_in_1)
        """
        # Calculate intersection area
        x_left = max(box1_params['x_min'], box2_params['x_min'])
        y_top = max(box1_params['y_min'], box2_params['y_min'])
        x_right = min(box1_params['x_max'], box2_params['x_max'])
        y_bottom = min(box1_params['y_max'], box2_params['y_max'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0, 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas
        box1_area = box1_params['width'] * box1_params['height']
        box2_area = box2_params['width'] * box2_params['height']

        if box1_area == 0 or box2_area == 0:
            return 0.0, 0.0

        iol_1_in_2 = intersection_area / box1_area
        iol_2_in_1 = intersection_area / box2_area

        return iol_1_in_2, iol_2_in_1

    def suppress_contained_boxes(self, boxes: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Step 0: Suppress Contained Boxes (Pre-processing)
        Remove smaller boxes that are almost entirely contained within larger ones.
        Returns filtered boxes and suppression statistics.
        """
        start_time = time.time()

        if not boxes:
            return [], {'boxes_suppressed': 0, 'processing_time': 0}

        print(f"  Step 0: Suppressing contained boxes from {len(boxes)} initial boxes...")

        n = len(boxes)
        boxes_to_suppress = set()
        suppression_reasons = {
            'contained_in_larger': 0,
            'total_pairs_checked': 0
        }

        # Check every unique pair of boxes
        for i in range(n):
            for j in range(i + 1, n):
                if i in boxes_to_suppress or j in boxes_to_suppress:
                    continue  # Skip if either box is already marked for suppression

                suppression_reasons['total_pairs_checked'] += 1
                box1_params = boxes[i]['bbox_params']
                box2_params = boxes[j]['bbox_params']

                # Calculate IoL for both directions
                iol_1_in_2, iol_2_in_1 = self.calculate_intersection_over_little(box1_params, box2_params)

                # If Box 1 is almost completely inside Box 2, suppress Box 1
                if iol_1_in_2 > self.contained_box_threshold:
                    boxes_to_suppress.add(i)
                    suppression_reasons['contained_in_larger'] += 1

                # If Box 2 is almost completely inside Box 1, suppress Box 2
                elif iol_2_in_1 > self.contained_box_threshold:
                    boxes_to_suppress.add(j)
                    suppression_reasons['contained_in_larger'] += 1

        # Create filtered list of non-suppressed boxes
        filtered_boxes = []
        for i, box in enumerate(boxes):
            if i not in boxes_to_suppress:
                filtered_boxes.append(box)

        processing_time = time.time() - start_time

        suppression_stats = {
            'boxes_before_suppression': len(boxes),
            'boxes_after_suppression': len(filtered_boxes),
            'boxes_suppressed': len(boxes_to_suppress),
            'suppression_rate': len(boxes_to_suppress) / len(boxes) if len(boxes) > 0 else 0,
            'processing_time': processing_time,
            'suppression_reasons': suppression_reasons
        }

        print(f"    Suppressed {len(boxes_to_suppress)} contained boxes")
        print(f"    Remaining boxes: {len(filtered_boxes)} (reduction: {suppression_stats['suppression_rate']:.1%})")
        print(f"    Pairs checked: {suppression_reasons['total_pairs_checked']}")
        print(f"    Processing time: {processing_time:.3f}s")

        return filtered_boxes, suppression_stats

    def extract_tile_number(self, tile_id: str) -> int:
        """Extract tile number from tile_id string like 'DURI-GENF05GS000-PRO-PID-IBU-0011-00_T0021' -> 21"""
        try:
            if '_T' in tile_id:
                return int(tile_id.split('_T')[-1])
            return 0
        except:
            return 0

    def merge_boxes(self, box1: Dict, box2: Dict) -> Dict:
        """Merge two boxes by taking the union of their bounding areas"""
        params1 = self.calculate_box_parameters(box1['global_bbox'])
        params2 = self.calculate_box_parameters(box2['global_bbox'])

        # Create merged bounding box
        merged_x_min = min(params1['x_min'], params2['x_min'])
        merged_y_min = min(params1['y_min'], params2['y_min'])
        merged_x_max = max(params1['x_max'], params2['x_max'])
        merged_y_max = max(params1['y_max'], params2['y_max'])

        merged_bbox = [
            [merged_x_min, merged_y_min],
            [merged_x_max, merged_y_min],
            [merged_x_max, merged_y_max],
            [merged_x_min, merged_y_max]
        ]

        # Create merged box data with aggregate metadata
        merged_box = {
            'global_bbox': merged_bbox,
            'bbox_params': self.calculate_box_parameters(merged_bbox)
        }

        # Aggregate tile information
        box1_tile_ids = box1.get('tile_ids', [box1.get('tile_id')])
        box2_tile_ids = box2.get('tile_ids', [box2.get('tile_id')])
        merged_box['tile_ids'] = list(set(box1_tile_ids + box2_tile_ids))

        box1_tile_numbers = box1.get('tile_numbers', [self.extract_tile_number(box1.get('tile_id', ''))])
        box2_tile_numbers = box2.get('tile_numbers', [self.extract_tile_number(box2.get('tile_id', ''))])
        merged_box['tile_numbers'] = sorted(list(set(box1_tile_numbers + box2_tile_numbers)))

        box1_tile_paths = box1.get('tile_paths', [box1.get('tile_path')])
        box2_tile_paths = box2.get('tile_paths', [box2.get('tile_path')])
        merged_box['tile_paths'] = list(set(box1_tile_paths + box2_tile_paths))

        box1_grid_positions = box1.get('grid_positions', [box1.get('grid_position')])
        box2_grid_positions = box2.get('grid_positions', [box2.get('grid_position')])
        all_positions = box1_grid_positions + box2_grid_positions
        # Remove duplicates while preserving list of lists structure
        unique_positions = []
        for pos in all_positions:
            if pos not in unique_positions and pos != "MERGED":
                unique_positions.append(pos)
        merged_box['grid_positions'] = unique_positions

        merged_box['source_tiles_count'] = len(merged_box['tile_ids'])

        # Keep consistent single-value metadata from box1
        merged_box['source_image'] = box1.get('source_image', '')
        merged_box['original_image_size'] = box1.get('original_image_size', [])
        merged_box['detection_type'] = box1.get('detection_type', 'ocr')
        merged_box['rotation_angle'] = box1.get('rotation_angle', 0)
        merged_box['tile_size'] = box1.get('tile_size', [])

        return merged_box

    def passes_primary_filter(self, box1_params: Dict, box2_params: Dict) -> Tuple[bool, str]:
        """
        Primary filter: Check orientation consistency and neighborhood filter.
        Returns (passes_filter, reason_for_rejection)
        """
        # Check orientation consistency
        box1_is_horizontal = box1_params['width'] > box1_params['height']
        box2_is_horizontal = box2_params['width'] > box2_params['height']

        if box1_is_horizontal != box2_is_horizontal:
            return False, "orientation_mismatch"

        # Neighborhood Filter
        # Identify the larger box based on area
        area1 = box1_params['width'] * box1_params['height']
        area2 = box2_params['width'] * box2_params['height']

        if area1 >= area2:
            larger_box = box1_params
            smaller_box = box2_params
        else:
            larger_box = box2_params
            smaller_box = box1_params

        # Create expanded search region around the larger box
        width_expansion = larger_box['width'] * self.neighborhood_expansion_factor / 2
        height_expansion = larger_box['height'] * self.neighborhood_expansion_factor / 2

        search_region = {
            'x_min': larger_box['x_min'] - width_expansion,
            'x_max': larger_box['x_max'] + width_expansion,
            'y_min': larger_box['y_min'] - height_expansion,
            'y_max': larger_box['y_max'] + height_expansion
        }

        # Check if the center of the smaller box falls inside the expanded search region
        smaller_center_x = smaller_box['center_x']
        smaller_center_y = smaller_box['center_y']

        is_neighbor = (search_region['x_min'] <= smaller_center_x <= search_region['x_max'] and
                      search_region['y_min'] <= smaller_center_y <= search_region['y_max'])

        if not is_neighbor:
            return False, "not_spatial_neighbors"

        return True, "passed"

    def should_connect_boxes_strict(self, box1_params: Dict, box2_params: Dict) -> Tuple[bool, str]:
        """
        Stricter connection logic with primary filters and precise connection rules.
        Returns (should_connect, connection_reason)
        """
        # Step 1: Apply Primary Filter
        passes_filter, filter_reason = self.passes_primary_filter(box1_params, box2_params)
        if not passes_filter:
            return False, f"primary_filter_failed_{filter_reason}"

        # Step 2: Check High Overlap (Rule A)
        iou = self.calculate_iou(box1_params, box2_params)
        if iou > self.high_overlap_threshold:
            return True, "high_overlap"

        # Keep existing high IoU threshold for merging identical detections
        if iou > self.high_iou_threshold:
            return True, "very_high_iou"

        # Step 3: Proximity Rules (Rule B)
        # Determine if boxes are horizontal or vertical
        is_horizontal = box1_params['width'] > box1_params['height']

        if is_horizontal:
            # For Horizontal Text
            # Condition 1: Vertical alignment (centers must be very close)
            vertical_center_distance = abs(box1_params['center_y'] - box2_params['center_y'])
            min_height = min(box1_params['height'], box2_params['height'])

            if vertical_center_distance >= min_height * self.vertical_alignment_factor:
                return False, "horizontal_text_poor_vertical_alignment"

            # Condition 2: Horizontal proximity (gap must not be too large)
            horizontal_gap = min(
                abs(box1_params['x_min'] - box2_params['x_max']),
                abs(box2_params['x_min'] - box1_params['x_max'])
            )
            max_height = max(box1_params['height'], box2_params['height'])

            if horizontal_gap < max_height * self.proximity_factor:
                return True, "horizontal_text_proximity"
            else:
                return False, "horizontal_text_too_distant"

        else:
            # For Vertical Text
            # Condition 1: Horizontal alignment (centers must be very close)
            horizontal_center_distance = abs(box1_params['center_x'] - box2_params['center_x'])
            min_width = min(box1_params['width'], box2_params['width'])

            if horizontal_center_distance >= min_width * self.horizontal_alignment_factor:
                return False, "vertical_text_poor_horizontal_alignment"

            # Condition 2: Vertical proximity (gap must not be too large)
            vertical_gap = min(
                abs(box1_params['y_min'] - box2_params['y_max']),
                abs(box2_params['y_min'] - box1_params['y_max'])
            )
            max_width = max(box1_params['width'], box2_params['width'])

            if vertical_gap < max_width * self.proximity_factor:
                return True, "vertical_text_proximity"
            else:
                return False, "vertical_text_too_distant"

        return False, "no_connection_rule_satisfied"

    def build_graph(self, boxes: List[Dict]) -> Tuple[Dict[int, Set[int]], Dict[str, int]]:
        """
        Build a graph where each box is a node and edges connect related boxes.
        Uses stricter connection criteria with neighborhood filter.
        Returns adjacency list representation and connection statistics.
        """
        if not boxes:
            return {}, {}

        graph = defaultdict(set)
        n = len(boxes)

        # Track connection reasons for metrics
        connection_stats = {
            'high_overlap_connections': 0,
            'very_high_iou_connections': 0,
            'horizontal_text_proximity_connections': 0,
            'vertical_text_proximity_connections': 0,
            'total_connections': 0,
            'primary_filter_rejections': 0,
            'orientation_mismatches': 0,
            'neighborhood_rejections': 0,
            'alignment_rejections': 0,
            'proximity_rejections': 0
        }

        # Create edges between boxes that should be connected
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    box1_params = boxes[i]['bbox_params']
                    box2_params = boxes[j]['bbox_params']

                    should_connect, reason = self.should_connect_boxes_strict(box1_params, box2_params)

                    if should_connect:
                        graph[i].add(j)
                        graph[j].add(i)
                        connection_stats['total_connections'] += 1

                        # Track connection reasons
                        if reason == "high_overlap":
                            connection_stats['high_overlap_connections'] += 1
                        elif reason == "very_high_iou":
                            connection_stats['very_high_iou_connections'] += 1
                        elif reason == "horizontal_text_proximity":
                            connection_stats['horizontal_text_proximity_connections'] += 1
                        elif reason == "vertical_text_proximity":
                            connection_stats['vertical_text_proximity_connections'] += 1
                    else:
                        # Track rejection reasons
                        if "primary_filter_failed" in reason:
                            connection_stats['primary_filter_rejections'] += 1
                            if "orientation_mismatch" in reason:
                                connection_stats['orientation_mismatches'] += 1
                            elif "not_spatial_neighbors" in reason:
                                connection_stats['neighborhood_rejections'] += 1
                        elif "alignment" in reason:
                            connection_stats['alignment_rejections'] += 1
                        elif "distant" in reason:
                            connection_stats['proximity_rejections'] += 1
                except Exception as e:
                    print(f"    Error processing box pair ({i}, {j}): {e}")
                    continue

        return graph, connection_stats

    def find_connected_components(self, graph: Dict[int, Set[int]], num_nodes: int) -> List[List[int]]:
        """
        Find all connected components in the graph using DFS.
        Returns list of clusters, where each cluster is a list of node indices.
        """
        visited = set()
        components = []

        for node in range(num_nodes):
            if node not in visited:
                # Start DFS from this unvisited node
                component = []
                stack = [node]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)

                        # Add all unvisited neighbors to stack
                        for neighbor in graph[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)

                components.append(component)

        return components

    def merge_cluster_boxes(self, cluster_indices: List[int], boxes: List[Dict]) -> Tuple[Dict, Dict[str, int]]:
        """
        Merge all boxes in a cluster into a single bounding box.
        For high IoU overlaps, keep the larger box instead of merging.
        Returns merged box and merge statistics.
        """
        merge_stats = {
            'original_cluster_size': len(cluster_indices),
            'boxes_filtered_out': 0,
            'final_cluster_size': 0,
            'merges_performed': 0
        }

        if not cluster_indices:
            # Return empty box if no indices
            empty_box = {
                'global_bbox': [[0, 0], [0, 0], [0, 0], [0, 0]],
                'bbox_params': self.calculate_box_parameters([[0, 0], [0, 0], [0, 0], [0, 0]]),
                'tile_ids': [],
                'source_image': '',
                'detection_type': 'empty'
            }
            return empty_box, merge_stats

        if len(cluster_indices) == 1:
            merge_stats['final_cluster_size'] = 1
            box_idx = cluster_indices[0]
            if box_idx < len(boxes):
                return boxes[box_idx], merge_stats
            else:
                # Index out of range, return empty box
                empty_box = {
                    'global_bbox': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'bbox_params': self.calculate_box_parameters([[0, 0], [0, 0], [0, 0], [0, 0]]),
                    'tile_ids': [],
                    'source_image': '',
                    'detection_type': 'empty'
                }
                return empty_box, merge_stats

        # Check for high IoU overlaps and filter to keep only the largest
        filtered_indices = []
        for i, idx1 in enumerate(cluster_indices):
            if idx1 >= len(boxes):
                continue

            should_keep = True
            for j, idx2 in enumerate(cluster_indices):
                if i != j and idx2 < len(boxes):
                    try:
                        iou = self.calculate_iou(boxes[idx1]['bbox_params'], boxes[idx2]['bbox_params'])
                        if iou > self.high_iou_threshold:
                            # Keep the larger box
                            area1 = boxes[idx1]['bbox_params']['width'] * boxes[idx1]['bbox_params']['height']
                            area2 = boxes[idx2]['bbox_params']['width'] * boxes[idx2]['bbox_params']['height']
                            if area1 < area2:
                                should_keep = False
                                break
                    except Exception as e:
                        print(f"      Error comparing boxes {idx1} and {idx2}: {e}")
                        continue

            if should_keep:
                filtered_indices.append(idx1)
            else:
                merge_stats['boxes_filtered_out'] += 1

        merge_stats['final_cluster_size'] = len(filtered_indices)

        # If after filtering we only have one box, return it
        if len(filtered_indices) == 1:
            return boxes[filtered_indices[0]], merge_stats

        if not filtered_indices:
            # No valid boxes after filtering
            empty_box = {
                'global_bbox': [[0, 0], [0, 0], [0, 0], [0, 0]],
                'bbox_params': self.calculate_box_parameters([[0, 0], [0, 0], [0, 0], [0, 0]]),
                'tile_ids': [],
                'source_image': '',
                'detection_type': 'empty'
            }
            return empty_box, merge_stats

        # Merge remaining boxes
        try:
            cluster_boxes = [boxes[idx] for idx in filtered_indices if idx < len(boxes)]
            if not cluster_boxes:
                empty_box = {
                    'global_bbox': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'bbox_params': self.calculate_box_parameters([[0, 0], [0, 0], [0, 0], [0, 0]]),
                    'tile_ids': [],
                    'source_image': '',
                    'detection_type': 'empty'
                }
                return empty_box, merge_stats

            merged_box = cluster_boxes[0]

            for box in cluster_boxes[1:]:
                merged_box = self.merge_boxes(merged_box, box)
                merge_stats['merges_performed'] += 1

            return merged_box, merge_stats
        except Exception as e:
            print(f"      Error during box merging: {e}")
            # Return the first box if merging fails
            if filtered_indices and filtered_indices[0] < len(boxes):
                return boxes[filtered_indices[0]], merge_stats
            else:
                empty_box = {
                    'global_bbox': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'bbox_params': self.calculate_box_parameters([[0, 0], [0, 0], [0, 0], [0, 0]]),
                    'tile_ids': [],
                    'source_image': '',
                    'detection_type': 'empty'
                }
                return empty_box, merge_stats

    def group_boxes_graph_based(self, boxes: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Group boxes using graph-based connected components approach with 5-step process:
        Step 0: Suppress contained boxes
        Step 1: Node creation
        Step 2: Edge creation with stricter criteria
        Step 3: Graph traversal to find clusters
        Step 4: Merge clusters
        """
        start_time = time.time()

        if not boxes:
            empty_stats = {
                'processing_time': 0,
                'boxes_before': 0,
                'boxes_after': 0,
                'suppression_stats': {'boxes_suppressed': 0, 'processing_time': 0},
                'components_found': 0,
                'graph_edges': 0,
                'isolated_nodes': 0,
                'connection_stats': {},
                'merge_stats': {'total_merges_performed': 0, 'total_boxes_filtered': 0, 'components_with_merges': 0},
                'component_size_distribution': {}
            }
            return [], empty_stats

        print(f"  Starting 5-step graph-based grouping for {len(boxes)} boxes...")

        # Step 0: Suppress Contained Boxes (Pre-processing)
        try:
            filtered_boxes, suppression_stats = self.suppress_contained_boxes(boxes)
        except Exception as e:
            print(f"  Error in Step 0 (suppression): {e}")
            # Return original boxes if suppression fails
            filtered_boxes = boxes
            suppression_stats = {'boxes_suppressed': 0, 'processing_time': 0}

        if not filtered_boxes:
            processing_time = time.time() - start_time
            final_stats = {
                'processing_time': processing_time,
                'boxes_before': len(boxes),
                'boxes_after': 0,
                'suppression_stats': suppression_stats,
                'components_found': 0,
                'graph_edges': 0,
                'isolated_nodes': 0,
                'connection_stats': {},
                'merge_stats': {'total_merges_performed': 0, 'total_boxes_filtered': 0, 'components_with_merges': 0},
                'component_size_distribution': {}
            }
            return [], final_stats

        # Step 1: Node Creation (implicit - each filtered box is a node)
        print(f"  Step 1: Created {len(filtered_boxes)} nodes from surviving boxes")

        # Step 2: Build graph of connections with stricter criteria
        print(f"  Step 2: Building graph with strict connection criteria...")
        try:
            graph, connection_stats = self.build_graph(filtered_boxes)
        except Exception as e:
            print(f"  Error in Step 2 (build graph): {e}")
            # Return individual boxes if graph building fails
            processing_time = time.time() - start_time
            final_stats = {
                'processing_time': processing_time,
                'boxes_before': len(boxes),
                'boxes_after': len(filtered_boxes),
                'suppression_stats': suppression_stats,
                'components_found': len(filtered_boxes),
                'graph_edges': 0,
                'isolated_nodes': len(filtered_boxes),
                'connection_stats': {},
                'merge_stats': {'total_merges_performed': 0, 'total_boxes_filtered': 0, 'components_with_merges': 0},
                'component_size_distribution': {1: len(filtered_boxes)}
            }
            return filtered_boxes, final_stats

        # Calculate graph metrics
        total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        nodes_with_connections = len([node for node, neighbors in graph.items() if neighbors])
        isolated_nodes = len(filtered_boxes) - nodes_with_connections

        print(f"    Graph built: {total_edges} edges, {nodes_with_connections} connected nodes, {isolated_nodes} isolated nodes")

        # Step 3: Find connected components
        print("  Step 3: Finding connected components...")
        try:
            components = self.find_connected_components(graph, len(filtered_boxes))
        except Exception as e:
            print(f"  Error in Step 3 (find components): {e}")
            # Treat each box as its own component
            components = [[i] for i in range(len(filtered_boxes))]

        # Analyze component sizes
        component_sizes = [len(comp) for comp in components]
        size_distribution = Counter(component_sizes)

        print(f"    Found {len(components)} components")
        print(f"    Component size distribution: {dict(size_distribution)}")

        # Step 4: Merge boxes in each component
        print("  Step 4: Merging boxes in components...")
        grouped_boxes = []
        total_merge_stats = {
            'total_merges_performed': 0,
            'total_boxes_filtered': 0,
            'components_with_merges': 0
        }

        for i, component in enumerate(components):
            try:
                merged_box, merge_stats = self.merge_cluster_boxes(component, filtered_boxes)
                grouped_boxes.append(merged_box)

                total_merge_stats['total_merges_performed'] += merge_stats['merges_performed']
                total_merge_stats['total_boxes_filtered'] += merge_stats['boxes_filtered_out']
                if merge_stats['merges_performed'] > 0:
                    total_merge_stats['components_with_merges'] += 1
            except Exception as e:
                print(f"    Error merging component {i}: {e}")
                # Add individual boxes from this component if merging fails
                for box_idx in component:
                    if box_idx < len(filtered_boxes):
                        grouped_boxes.append(filtered_boxes[box_idx])

        processing_time = time.time() - start_time

        # Compile final statistics
        final_stats = {
            'processing_time': processing_time,
            'boxes_before': len(boxes),
            'boxes_after': len(grouped_boxes),
            'reduction_ratio': (len(boxes) - len(grouped_boxes)) / len(boxes) if len(boxes) > 0 else 0,
            'suppression_stats': suppression_stats,
            'components_found': len(components),
            'graph_edges': total_edges,
            'isolated_nodes': isolated_nodes,
            'connection_stats': connection_stats,
            'merge_stats': total_merge_stats,
            'component_size_distribution': dict(size_distribution)
        }

        print(f"  5-step grouping completed in {processing_time:.3f}s")
        print(f"  📊 Final Results: {len(boxes)} → {len(grouped_boxes)} boxes (reduction: {final_stats['reduction_ratio']:.1%})")

        return grouped_boxes, final_stats

    def process_image_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all JSON files in an image folder using graph-based grouping"""
        folder_start_time = time.time()
        folder_name = os.path.basename(folder_path)

        print(f"\n=== Processing folder: {folder_name} ===")

        # Load all JSON data
        all_data = self.load_json_files(folder_path)

        if not all_data:
            print(f"  No data found in {folder_path}")
            return []

        print(f"  Loaded {len(all_data)} detection boxes from JSON files")

        # Convert to global coordinates and calculate parameters
        processed_boxes = []
        failed_conversions = 0

        for i, item in enumerate(all_data):
            try:
                # Validate required fields
                if 'bbox' not in item:
                    print(f"    Warning: Item {i} missing 'bbox' field")
                    failed_conversions += 1
                    continue

                if 'tile_coordinates' not in item:
                    print(f"    Warning: Item {i} missing 'tile_coordinates' field")
                    failed_conversions += 1
                    continue

                # Convert coordinates
                global_bbox = self.convert_to_global_coordinates(item['bbox'], item['tile_coordinates'])
                bbox_params = self.calculate_box_parameters(global_bbox)

                processed_item = item.copy()
                processed_item['global_bbox'] = global_bbox
                processed_item['bbox_params'] = bbox_params

                # Initialize list fields for individual boxes with safe defaults
                processed_item['tile_ids'] = [item.get('tile_id', 'UNKNOWN')]
                processed_item['tile_numbers'] = [self.extract_tile_number(item.get('tile_id', ''))]
                processed_item['tile_paths'] = [item.get('tile_path', '')]
                processed_item['grid_positions'] = [item.get('grid_position', [])]
                processed_item['source_tiles_count'] = 1

                processed_boxes.append(processed_item)

            except Exception as e:
                print(f"    Error processing item {i}: {e}")
                print(f"    Item data: {item}")
                failed_conversions += 1
                continue

        if failed_conversions > 0:
            print(f"  Warning: Failed to process {failed_conversions} items")

        if not processed_boxes:
            print(f"  No valid boxes found after preprocessing")
            return []

        print(f"  Preprocessed {len(processed_boxes)} boxes for grouping (failed: {failed_conversions})")

        # Apply graph-based grouping (works for all orientations simultaneously)
        try:
            grouped_boxes, grouping_stats = self.group_boxes_graph_based(processed_boxes)
        except Exception as e:
            print(f"  Error in grouping: {e}")
            return []

        # Update global metrics
        self.metrics['total_boxes_before_grouping'] += grouping_stats['boxes_before']
        self.metrics['total_boxes_after_grouping'] += grouping_stats['boxes_after']
        self.metrics['total_merges_performed'] += grouping_stats['merge_stats']['total_merges_performed']
        self.metrics['processing_times'].append(grouping_stats['processing_time'])

        # Clean up the output format
        output_data = []
        for box in grouped_boxes:
            try:
                global_bbox = box['global_bbox']

                # Calculate normalized bbox relative to original image size
                original_width, original_height = box.get('original_image_size', [1, 1])
                if original_width == 0 or original_height == 0:
                    original_width, original_height = 1, 1

                bbox_normalized = [
                    [point[0] / original_width, point[1] / original_height]
                    for point in global_bbox
                ]

                # Create clean output box with global coordinates as the main bbox
                clean_box = {
                    'bbox': global_bbox,
                    'bbox_normalized': bbox_normalized,
                    'rotation_angle': box.get('rotation_angle', 0),
                    'source_image': box.get('source_image', ''),
                    'detection_type': box.get('detection_type', 'craft_detection'),
                    'original_image_size': box.get('original_image_size', []),
                    'grouped_from_tiles': box.get('tile_ids', [box.get('tile_id', 'UNKNOWN')]),
                    'is_grouped': len(box.get('tile_ids', [])) > 1
                }

                output_data.append(clean_box)

            except Exception as e:
                print(f"    Error creating output box: {e}")
                continue

        folder_processing_time = time.time() - folder_start_time
        print(f"\n  📊 Folder Summary for {folder_name}:")
        print(f"     Total processing time: {folder_processing_time:.3f}s")
        print(f"     Input boxes: {len(all_data)}")
        print(f"     Valid processed boxes: {len(processed_boxes)}")
        print(f"     Output groups: {len(output_data)}")
        if len(all_data) > 0:
            print(f"     Reduction ratio: {((len(all_data) - len(output_data)) / len(all_data)):.1%}")
        print(f"     Components found: {grouping_stats['components_found']}")
        print(f"     Graph edges created: {grouping_stats['graph_edges']}")

        return output_data

    def process_all_images(self):
        """Process all image folders in the detection directory"""
        total_start_time = time.time()

        if not os.path.exists(self.detection_dir):
            print(f"Detection directory not found: {self.detection_dir}")
            return

        # Get all subdirectories
        subdirs = [d for d in os.listdir(self.detection_dir)
                   if os.path.isdir(os.path.join(self.detection_dir, d))]

        print(f"\n🚀 Starting batch processing of {len(subdirs)} image folders...")
        print(f"Detection directory: {self.detection_dir}")
        print(f"Output directory: {self.group_dir}")

        successful_processes = 0
        failed_processes = 0

        for i, subdir in enumerate(subdirs, 1):
            folder_path = os.path.join(self.detection_dir, subdir)
            print(f"\n[{i}/{len(subdirs)}] Processing folder: {subdir}")

            try:
                # Process the folder
                grouped_data = self.process_image_folder(folder_path)

                # Save results
                output_filename = f"{subdir}_grouped.json"
                output_path = os.path.join(self.group_dir, output_filename)

                with open(output_path, 'w') as f:
                    json.dump(grouped_data, f, indent=2)
                print(f"  ✅ Saved grouped data to: {output_filename}")
                successful_processes += 1

            except Exception as e:
                print(f"  ❌ Error processing {subdir}: {e}")
                failed_processes += 1

        # Update global metrics
        self.metrics['total_images_processed'] = successful_processes

        # Print final summary
        total_processing_time = time.time() - total_start_time

        print(f"\n" + "="*60)
        print(f"🎯 BATCH PROCESSING COMPLETE")
        print(f"="*60)
        print(f"Total processing time: {total_processing_time:.2f}s")
        print(f"Successfully processed: {successful_processes}/{len(subdirs)} folders")
        print(f"Failed processes: {failed_processes}")
        print(f"\n📈 GLOBAL METRICS:")
        print(f"   Total boxes before grouping: {self.metrics['total_boxes_before_grouping']:,}")
        print(f"   Total boxes after grouping: {self.metrics['total_boxes_after_grouping']:,}")
        print(f"   Total reduction: {self.metrics['total_boxes_before_grouping'] - self.metrics['total_boxes_after_grouping']:,}")
        print(f"   Overall reduction ratio: {((self.metrics['total_boxes_before_grouping'] - self.metrics['total_boxes_after_grouping']) / self.metrics['total_boxes_before_grouping']):.1%}")
        print(f"   Total merges performed: {self.metrics['total_merges_performed']:,}")

        if self.metrics['processing_times']:
            avg_time = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            print(f"   Average grouping time per folder: {avg_time:.3f}s")
            print(f"   Fastest grouping: {min(self.metrics['processing_times']):.3f}s")
            print(f"   Slowest grouping: {max(self.metrics['processing_times']):.3f}s")

        print(f"\n✨ All results saved to: {self.group_dir}")

if __name__ == "__main__":
    grouper = BoundingBoxGrouper()
    grouper.process_all_images()

#


