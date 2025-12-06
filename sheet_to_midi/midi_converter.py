"""
violin_extractor.py — Generalized Multi-Format Violin Extraction
Makes violin-only extraction work for various score types:
- Solo violin scores - extracts all staffs
- Chamber music scores - extracts only violin parts (1st and 2nd violin)
- Full orchestral scores - extracts ONLY Violin 1 parts
- Handles different page layouts and formats
- Improved OCR for violin part detection
- Robust staff detection algorithm
"""

import os, sys, math, cv2, pytesseract, numpy as np, argparse, logging
from random import randint
from typing import List, Tuple, Dict, Optional
from midiutil import MIDIFile

# Configure logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

# Constants
LEFT_CROP_PX = 20  # Default left crop to remove binding/margin
STATIC_FOLDER = "static"
# Expanded violin keyword list
VIOLIN_KEYWORDS = [
    'violin', 'vln', 'vln.', 'vln1', 'vln2', 'violin i', 'violin ii', 
    'vn', 'vn.', 'vno', 'vno.', 'violino', '1st violin', '2nd violin',
    'first violin', 'second violin', 'solo violin', 'violino primo',
    'violino secondo', 'violins', 'v.', 'violin solo'
]
# Any instrument name for orchestral detection
INSTRUMENT_KEYWORDS = [
    'flute', 'oboe', 'clarinet', 'bassoon', 'horn', 'trumpet', 'trombone', 'tuba',
    'timpani', 'percussion', 'harp', 'piano', 'violin', 'viola', 'cello', 'bass',
    'soprano', 'alto', 'tenor', 'baritone', 'guitar', 'organ', 'harpsichord',
    'fl.', 'ob.', 'cl.', 'bsn.', 'hn.', 'tpt.', 'tbn.', 'vla.', 'vc.', 'cb.',
    'picc.', 'cor.', 'fagot', 'timp.'
]
# Staff detection parameters
STAFF_HEIGHT_RATIO_MIN = 0.01  # Min staff height relative to image height
STAFF_HEIGHT_RATIO_MAX = 0.08   # Max staff height relative to image height
LINE_SPACING_TOLERANCE = 0.5   # Acceptable line spacing variation within staff
# OCR parameters
OCR_CONFIDENCE_THRESHOLD = 30  # Minimum OCR confidence

class BBox:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.cx, self.cy = x + w / 2, y + h / 2
    
    def overlap(self, o):
        ox = max(0, min(self.x + self.w, o.x + o.w) - max(self.x, o.x))
        oy = max(0, min(self.y + self.h, o.y + o.h) - max(self.y, o.y))
        return (ox * oy) / (self.w * self.h)
    
    def dist(self, o): 
        return math.hypot(self.cx - o.cx, self.cy - o.cy)
    
    def merge(self, o):
        x0, y0 = min(self.x, o.x), min(self.y, o.y)
        x1, y1 = max(self.x + self.w, o.x + o.w), max(self.y + self.h, o.y + o.h)
        return BBox(x0, y0, x1 - x0, y1 - y0)
    
    def draw(self, img, col, th=2):
        cv2.rectangle(img, (int(self.x), int(self.y)), (int(self.x + self.w), int(self.y + self.h)), col, th)

    def __repr__(self):
        return f"BBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

# Note mapping for MIDI conversion
_NOTE = {i: (n, p) for i, (n, p) in enumerate([
    ('c5', 72), ('b4', 71), ('a4', 69), ('g4', 67), ('f4', 65), ('e4', 64), ('d4', 62), ('c4', 60),
    ('b3', 59), ('a3', 57), ('g3', 55), ('f3', 53), ('e3', 52), ('d3', 50), ('c3', 48), ('b2', 47), ('a2', 45)
])}
_STEP = 0.0625

class NoteEvt:
    def __init__(self, box, Btype, staff, sharps, flats):
        idx = int(((box.cy - staff.y) / staff.h) / _STEP + 0.5)
        name, pitch = _NOTE.get(idx, ('c4', 60))
        if any(s for s in sharps if s.note[0] == name[0]): 
            name += '#'
            pitch += 1
        if any(f for f in flats if f.note[0] == name[0]): 
            name += 'b'
            pitch -= 1
        self.note, self.pitch, self.dur, self.box = name, pitch, {"1": 4, "2": 2, "4,8": 1}.get(Btype, 1), box

class ScoreAnalyzer:
    def __init__(self, img_path, debug=False):
        """Initialize with an image path and load the score"""
        self.debug = debug
        self.img_path = img_path
        self.gray = cv2.imread(img_path, 0)
        if self.gray is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Score properties
        h, w = self.gray.shape
        self.original_size = (w, h)
        # We'll properly detect orchestral format later, not just by ratio
        self.is_orchestral = False  # Will be detected based on instrument names
        logging.info(f"Image size: {w}x{h}, ratio: {w/h:.2f}")
        
        # Initialize detection results
        self.staffs = []
        self.violin_staffs = []
        self.all_staffs = []  # Store all staffs for solo/chamber scores
        self.instrument_positions = []  # All instrument names found
        self.violin_positions = []
        self.notes = []
        self.debug_images = {}
    
    def preprocess(self, left_crop=LEFT_CROP_PX, auto_crop=True):
        """Preprocess the image with enhancements"""
        # Detect left margin size automatically if requested
        if auto_crop:
            left_crop = self._detect_left_margin()
            logging.info(f"Auto-detected left margin: {left_crop}px")
        
        # Store the original image before cropping for instrument detection
        self.original_gray = self.gray.copy()
        self.left_margin = self.gray[:, :left_crop+5]  # Keep a little extra for OCR
        
        # Crop left margin
        self.gray = self.gray[:, left_crop:]
        
        # Apply image enhancements
        self.enhanced = self._enhance_image(self.gray)
        _, self.binary = cv2.threshold(self.enhanced, 127, 255, cv2.THRESH_BINARY)
        _, self.binary_inv = cv2.threshold(self.enhanced, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Create debug image
        self.debug_img = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        
        # IMPORTANT: Detect if this is an orchestral score by looking for instrument names
        self._detect_score_type()
        
        return self
    
    def _detect_score_type(self):
        """Detect if this is an orchestral score based on instrument names in the margin"""
        # Enhance the left margin for better OCR
        margin_enhanced = self._enhance_image(self.left_margin)
        _, margin_binary = cv2.threshold(margin_enhanced, 127, 255, cv2.THRESH_BINARY)
        
        # Try multiple OCR configurations to find instrument names
        ocr_configs = [
            '--psm 6 -l eng',  # Block of text
            '--psm 4 -l eng',  # Single column of text
            '--psm 7 -l eng',  # Single line
            '--psm 8 -l eng',  # Single word
            '--psm 3 -l eng'   # Auto page segmentation
        ]
        
        found_instruments = []
        non_violin_instruments = []
        violin1_positions = []  # Specifically for "Violin 1" or "Violin I"

        # Try multiple OCR configurations
        for config in ocr_configs:
            data = pytesseract.image_to_data(margin_binary, config=config, 
                                            output_type=pytesseract.Output.DICT)
            
            for text, top, height, conf in zip(data['text'], data['top'], data['height'], data['conf']):
                if conf > OCR_CONFIDENCE_THRESHOLD:
                    text = text.lower().strip()
                    center_y = top + height / 2

                    violin_keyword_hit = any(vk in text for vk in VIOLIN_KEYWORDS)

                    if violin_keyword_hit:
                        if any(bad in text for bad in ['2', 'second', 'viola']):
                            logging.info(f"Excluded '{text}' (looks like Violin 2 or second violin)")
                            continue  

                        found_instruments.append((text, center_y))
                        logging.info(f"OCR found violin instrument '{text}' at y={center_y}")

                        if any(v1k in text for v1k in ['violin 1', 'violin i', 'violin1', 'violino primo', '1st violin', 'first violin', 'vln 1', 'vln i', 'vln1', 'vln. 1', 'vln.1', 'vln. i']):
                            violin1_positions.append(center_y)
                            logging.info(f"Detected Violin 1 at y={center_y}")
                        
                        self.violin_positions.append(center_y)

                    else:
                        for keyword in INSTRUMENT_KEYWORDS:
                            if keyword in text:
                                non_violin_instruments.append((text, center_y))
                                logging.info(f"OCR found non-violin instrument '{text}' at y={center_y}")
                                break

        # Correct score type classification
        self.is_chamber_ensemble = False
        self.is_full_orchestra = False
        self.is_orchestral = False

        if found_instruments:
            violin_found = any('violin' in name for name, _ in found_instruments)
            non_violin_found = any(name not in VIOLIN_KEYWORDS for name, _ in found_instruments)
            
            if violin_found and non_violin_found:
                if len(found_instruments) >= 5:
                    self.is_full_orchestra = True
                    logging.info("Full orchestra detected with multiple instrument sections")
                else:
                    self.is_chamber_ensemble = True
                    logging.info("Chamber ensemble detected with violin and other instruments")
            elif violin_found and not non_violin_found:
                self.is_chamber_ensemble = True
                logging.info("Violin-only chamber ensemble detected")
            else:
                self.is_orchestral = True
                logging.info("General orchestral score detected")

        logging.info(f"Score type detection: {'full orchestra' if self.is_full_orchestra else 'chamber ensemble' if self.is_chamber_ensemble else 'orchestral' if self.is_orchestral else 'solo/violin'}")

        # Save the instruments found for later use
        self.instrument_positions = [(name, pos) for name, pos in found_instruments]

        # Mark specific violin positions
        for name, pos in self.instrument_positions:
            if any(vk in name for vk in VIOLIN_KEYWORDS):
                self.violin_positions.append(pos)

        # Store violin1 positions for full orchestra mode
        self.violin1_positions = violin1_positions

        # If no instruments were found, try sectional OCR
        if not self.instrument_positions:
            self._try_sectional_ocr()
    
    def _try_sectional_ocr(self):
        """Try sectional OCR for instrument detection"""
        h, _ = self.left_margin.shape
        sections = 20
        section_height = h // sections
        
        # Enhance the left margin for better OCR
        margin_enhanced = self._enhance_image(self.left_margin)
        _, margin_binary = cv2.threshold(margin_enhanced, 127, 255, cv2.THRESH_BINARY)
        
        for i in range(sections):
            y_start = i * section_height
            y_end = (i + 1) * section_height
            section = margin_binary[y_start:y_end, :]
            
            # Try with different PSM modes
            for config in ['--psm 7 -l eng', '--psm 8 -l eng']:
                text = pytesseract.image_to_string(section, config=config).lower().strip()
                
                # Check for any instrument
                for keyword in INSTRUMENT_KEYWORDS:
                    if keyword in text:
                        center_y = y_start + section_height / 2
                        self.instrument_positions.append((keyword, center_y))
                        logging.info(f"Section scan found instrument '{keyword}' at y={center_y}")
                        
                        # Also check if it's a violin
                        if any(vk in text for vk in VIOLIN_KEYWORDS):
                            self.violin_positions.append(center_y)
                            logging.info(f"Found violin at y={center_y}")
                            
                            # Check specifically for Violin 1
                            for v1k in ['violin 1', 'violin i', 'violin1', 'violino primo', '1st violin', 'first violin', 'vln 1', 'vln i', 'vln1', 'vln. 1', 'vln.1', 'vln. i']:
                                if v1k in text:
                                    if hasattr(self, 'violin1_positions'):
                                        self.violin1_positions.append(center_y)
                                    else:
                                        self.violin1_positions = [center_y]
                                    logging.info(f"Found Violin 1 at y={center_y}")
                                    break
                        
                        # Update orchestral detection
                        self.is_orchestral = True
                        
                        # Update full orchestra detection if we find many instruments
                        if len(self.instrument_positions) >= 5:
                            self.is_full_orchestra = True
                        break
    
    def _detect_left_margin(self):
        """Automatically detect the left margin size"""
        h, w = self.gray.shape
        # Compute column-wise pixel density
        col_density = np.sum(self.gray < 127, axis=0) / h
        
        # Smooth the density profile
        window = max(5, w // 100)
        kernel = np.ones(window) / window
        smoothed = np.convolve(col_density, kernel, mode='same')
        
        # Find the first significant increase in density
        threshold = 0.05
        baseline = np.mean(smoothed[:10])
        
        for x in range(10, min(w // 3, 200)):  # Look in first third of image, max 200px
            if smoothed[x] > baseline + threshold:
                return max(0, x - 5)  # Subtract a small buffer
        
        return LEFT_CROP_PX  # Default if no clear margin detected
    
    def _enhance_image(self, gray):
        """Apply image enhancements for better feature detection"""
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def detect_staffs(self):
        """Multi-strategy staff detection"""
        h, w = self.gray.shape
        min_staff_height = int(h * STAFF_HEIGHT_RATIO_MIN)
        max_staff_height = int(h * STAFF_HEIGHT_RATIO_MAX)
        
        # Strategy 1: Template matching
        if self.debug:
            logging.info("Attempting template-based staff detection...")
        staffs_template = self._template_match_staffs()
        
        # Strategy 2: Hough line detection
        if self.debug:
            logging.info("Attempting Hough line staff detection...")
        staffs_hough = self._hough_detect_staffs()
        
        # Strategy 3: Horizontal projection
        if self.debug:
            logging.info("Attempting projection-based staff detection...")
        staffs_projection = self._projection_detect_staffs()
        
        # Combine results from different methods
        all_staffs = staffs_template + staffs_hough + staffs_projection
        
        # Filter by size constraints
        filtered_staffs = [s for s in all_staffs if min_staff_height <= s.h <= max_staff_height]
        
        # Merge overlapping staff regions
        self.staffs = self._merge_regions(filtered_staffs, 0.3)
        
        if not self.staffs and staffs_hough:
            # Fallback to unfiltered Hough staffs if nothing passes filters
            self.staffs = staffs_hough
            logging.warning("Using unfiltered Hough staffs as fallback")
        
        # Store all detected staffs for solo/chamber scores
        self.all_staffs = sorted(self.staffs, key=lambda s: s.y)
        
        logging.info(f"Total staff lines detected: {len(self.staffs)}")
        
        # Draw staffs on debug image
        for i, staff in enumerate(self.staffs):
            color = (0, 0, 255) if i % 3 == 0 else (0, 255, 0) if i % 3 == 1 else (255, 0, 0)
            staff.draw(self.debug_img, color, 1)
            cv2.putText(self.debug_img, f"#{i}", (int(staff.x) + 10, int(staff.cy)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        self.debug_images["staffs"] = self.debug_img.copy()
        return self
    
    def _template_match_staffs(self):
        """Detect staffs using template matching"""
        # This would require pre-made templates
        # For the generic solution, we'll return an empty list
        return []
    
    def _hough_detect_staffs(self):
        """Detect staffs using Hough line transform with retries"""
        h, w = self.binary_inv.shape

        # Retry settings
        retry_params = [
            {"kernel_div": 30, "canny1": 50, "canny2": 150},
            {"kernel_div": 60, "canny1": 30, "canny2": 100},
            {"kernel_div": 80, "canny1": 20, "canny2": 80},
        ]
        
        for attempt, params in enumerate(retry_params):
            # Morphological processing to enhance horizontal lines
            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (w // params["kernel_div"], 1)
            )
            horizontal = cv2.morphologyEx(self.binary_inv, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Detect edges
            edges = cv2.Canny(horizontal, params["canny1"], params["canny2"])
            
            # Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=w*0.3, maxLineGap=20)
            
            if lines is not None:
                # Extract y-coordinates of horizontal lines
                y_coords = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y2 - y1) < 10:  # horizontal line
                        y_coords.append((y1 + y2) / 2)
                
                # Remove duplicates
                y_coords.sort()
                filtered_coords = []
                for y in y_coords:
                    if not filtered_coords or abs(y - filtered_coords[-1]) > 5:
                        filtered_coords.append(y)
                
                # Group into 5-line staffs
                staffs = []
                i = 0
                while i <= len(filtered_coords) - 5:
                    group = filtered_coords[i:i+5]
                    gaps = [group[j+1] - group[j] for j in range(4)]
                    avg_gap = sum(gaps) / 4

                    if all(abs(gap - avg_gap) < avg_gap * LINE_SPACING_TOLERANCE for gap in gaps):
                        y_min = group[0]
                        y_max = group[4]
                        staff_height = y_max - y_min
                        margin = staff_height * 0.4
                        staffs.append(BBox(0, int(y_min - margin), w, int(staff_height + 2 * margin)))
                        i += 5
                    else:
                        i += 1
                
                if staffs:
                    logging.info(f"Hough detection succeeded on attempt {attempt+1}")
                    return staffs
            
            logging.warning(f"Hough detection attempt {attempt+1} failed, retrying...")
        
        # All attempts failed
        logging.error("Staff detection failed after retries.")
        return []
    
    def _projection_detect_staffs(self):
        """Detect staffs using horizontal projection profile"""
        h, w = self.binary_inv.shape
        
        # Calculate horizontal projection
        projection = np.sum(self.binary_inv, axis=1)
        
        # Smooth projection
        kernel_size = max(3, h // 100)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(projection, kernel, mode='same')
        
        # Find peaks (staff line positions)
        peak_threshold = np.mean(smoothed) * 1.0
        peak_indices = []
        
        for i in range(1, h-1):
            if smoothed[i] > peak_threshold and smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                peak_indices.append(i)
        
        # Group peaks into staffs (5 lines per staff)
        staffs = []
        i = 0
        while i <= len(peak_indices) - 5:
            group = peak_indices[i:i+5]
            gaps = [group[j+1] - group[j] for j in range(4)]
            avg_gap = sum(gaps) / 4
            
            # Check if gaps are consistent
            if all(abs(gap - avg_gap) < avg_gap * LINE_SPACING_TOLERANCE for gap in gaps):
                y_min = group[0]
                y_max = group[4]
                staff_height = y_max - y_min
                
                # Add some margin above and below
                margin = staff_height * 0.4
                staffs.append(BBox(0, int(y_min - margin), w, int(staff_height + 2*margin)))
                i += 5
            else:
                i += 1
        
        return staffs
    
    def _merge_regions(self, regions, overlap_threshold=0.3):
        """Merge overlapping regions"""
        if not regions:
            return []
        
        regions = regions.copy()
        merged = []
        
        while regions:
            r = regions.pop(0)
            regions.sort(key=lambda x: r.dist(x))
            changed = True
            
            while changed:
                changed = False
                i = 0
                while i < len(regions):
                    if r.overlap(regions[i]) > overlap_threshold or regions[i].overlap(r) > overlap_threshold:
                        r = r.merge(regions.pop(i))
                        changed = True
                    else:
                        i += 1
            
            merged.append(r)
        
        return merged
    
    def detect_violin_parts_force_all(self):
        """Force use of all staffs regardless of OCR detection."""
        logging.info("Forcing use of all staffs regardless of violin label detection.")
        self.violin_staffs = self.staffs
        return self

    def detect_violin_parts(self):
        """Detect violin parts by scanning the left margin for violin labels (full-page OCR)."""
        if not self.staffs:
            logging.error("No staffs detected, cannot find violin parts")
            return self

        h, w = self.original_size
        margin_width_candidates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        violin_positions = []

        ocr_configs = [
            '--psm 6 -l eng',  # Block
            '--psm 4 -l eng',  # Column
            '--psm 7 -l eng',  # Line
            '--psm 8 -l eng',  # Word
            '--psm 3 -l eng'   # Auto
        ]

        # Try different margin widths
        for frac in margin_width_candidates:
            margin_width = int(w * frac)
            margin = self.original_gray[:, :margin_width]

            # Enhance margin
            margin_enhanced = self._enhance_image(margin)
            _, margin_binary = cv2.threshold(margin_enhanced, 127, 255, cv2.THRESH_BINARY)

            for config in ocr_configs:
                data = pytesseract.image_to_data(margin_binary, config=config, output_type=pytesseract.Output.DICT)

                for text, top, height, conf in zip(data['text'], data['top'], data['height'], data['conf']):
                    if conf > OCR_CONFIDENCE_THRESHOLD:
                        text = text.lower().strip()
                        if any(vk in text for vk in VIOLIN_KEYWORDS):
                            if "viola" in text:
                                logging.info(f"Excluded '{text}' (looks like viola, not violin)")
                                continue
                            center_y = top + height / 2
                            violin_positions.append(center_y)
                            logging.info(f"OCR found '{text}' at y={center_y} with margin {int(frac*100)}%")
            
            if violin_positions:
                break  # Found at this margin, no need to expand more

        # 🔹 Case 1: No OCR match at all — no instrument labels
        if not violin_positions and not self.instrument_positions:
            logging.warning("No instrument labels found. Using all staffs.")
            self.violin_staffs = self.staffs
            return self

        # 🔸 Case 2: Violin label not found but other instruments exist
        if not violin_positions:
            logging.error("No violin-related text detected, cannot find violin parts after margin expansion.")
            return self

        # if not violin_positions:
        #     if not self.instrument_positions:
        #         logging.warning("No instrument names detected, defaulting to all staffs for processing.")
        #         self.violin_staffs = self.all_staffs
        #         return self
        #     else:
        #         logging.error("No violin-related text detected, cannot find violin parts after margin expansion.")
        #         return self

        # Match violin positions to staffs
        matched_staffs = []
        sorted_staffs = sorted(self.staffs, key=lambda s: s.y)

        for v_pos in violin_positions:
            nearest_staffs = sorted(sorted_staffs, key=lambda s: abs(s.cy - v_pos))
            if nearest_staffs:
                nearest = nearest_staffs[0]
                if abs(nearest.cy - v_pos) < nearest.h * 2.0:  # Reasonable threshold
                    matched_staffs.append(nearest)

        if not matched_staffs:
            logging.error("Violin text found, but could not match to any staff.")
            return self

        self.violin_staffs = list({s for s in matched_staffs})  # Deduplicate

        # Highlight on debug image
        for staff in self.violin_staffs:
            staff.draw(self.debug_img, (0, 255, 0), 2)

        self.debug_images["violin_parts"] = self.debug_img.copy()
        return self

    def _map_violin1_parts_in_orchestra(self):
        """Map specifically Violin 1 parts in a full orchestra score"""
        if not self.staffs:
            return []
            
        sorted_staffs = sorted(self.staffs, key=lambda s: s.y)
        violin1_staffs = []
        
        # If we have Violin 1 positions, use those
        if hasattr(self, 'violin1_positions') and self.violin1_positions:
            logging.info(f"Found {len(self.violin1_positions)} Violin 1 positions in orchestra")
            
            # Map Violin 1 positions to nearest staffs
            for v1_pos in self.violin1_positions:
                # Find the closest staff to this violin 1 position
                closest_staff = min(sorted_staffs, key=lambda s: abs(s.cy - v1_pos))
                max_distance = closest_staff.h * 3.0  # Allow reasonable distance
                
                if abs(closest_staff.cy - v1_pos) < max_distance:
                    violin1_staffs.append(closest_staff)
                    logging.info(f"Matched Violin 1 position at y={v1_pos} to staff at y={closest_staff.cy}")
        
        # If direct mapping didn't work, use typical orchestra layout
        if not violin1_staffs:
            logging.info("Using typical orchestra layout to identify Violin 1")
            
            # For orchestral scores, if instrument names are on the left
            # Violin 1 is typically near the top of the score, after flutes/woodwinds
            if len(sorted_staffs) >= 10:  # Full orchestra
                # Typically, 1st violins are the 3rd staff in a full orchestra score
                # (after flute and oboe)
                violin1_staffs = [sorted_staffs[2]]
            elif len(sorted_staffs) >= 5:  # Smaller orchestra 
                # In smaller scores, 1st violins might be the 1st or 2nd staff
                violin1_staffs = [sorted_staffs[0]]
        
        # If we still found nothing and this is a string orchestra,
        # the first staff is almost certainly Violin 1
        if not violin1_staffs and len(sorted_staffs) >= 3:
            logging.info("Assuming first staff is Violin 1 in string orchestra")
            violin1_staffs = [sorted_staffs[0]]
            
        # Always limit to just one staff - we only want Violin 1
        if len(violin1_staffs) > 1:
            logging.info(f"Found {len(violin1_staffs)} Violin 1 staffs, selecting only the top one")
            violin1_staffs = [min(violin1_staffs, key=lambda s: s.y)]
            
        return violin1_staffs
    
    
    def detect_notes(self, tmpl_dir='resources/template'):
        """Detect notes on the selected violin/all staffs"""
        if not self.violin_staffs:
            logging.error("No staffs identified for note detection")
            return self
        
        # Template matching for musical symbols
        templates = {
            'sharp': ['sharp.png'],
            'flat': ['flat-line.png', 'flat-space.png'],
            'quarter': ['quarter.png', 'solid-note.png'],
            'half': ['half-space.png', 'half-note-line.png', 'half-line.png', 'half-note-space.png'],
            'whole': ['whole-space.png', 'whole-note-line.png', 'whole-line.png', 'whole-note-space.png'],
            'sixteenth': ['sixteenth-note.png', 'special-sixteenth.png'],
            'beam': ['beam-group.png']   
        }
        
        # Load templates and detect symbols
        detected = {}
        for symbol, file_list in templates.items():
            try:
                templates_loaded = self._load_templates(tmpl_dir, file_list)
                detected[symbol] = self._detect_symbols(self.binary, templates_loaded)
            except Exception as e:
                logging.warning(f"Error detecting {symbol}: {e}")
                detected[symbol] = []
        
        # Create debug visualization
        notes_debug = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)
        
        # Process detected symbols for each selected staff
        self.notes = []
        for staff in self.violin_staffs:
            staff_notes = []
            
            # Helper to filter symbols within this staff
            in_staff = lambda b: staff.y - staff.h * 0.1 <= b.cy <= staff.y + staff.h * 1.1
            
            # Collect accidentals
            sharps = [NoteEvt(b, 'sharp', staff, [], []) for b in detected.get('sharp', []) if in_staff(b)]
            flats = [NoteEvt(b, 'flat', staff, [], []) for b in detected.get('flat', []) if in_staff(b)]
            
            # Collect notes with appropriate durations
            for note_type, boxes, duration in [
                ('quarter', detected.get('quarter', []), '4,8'),
                ('half', detected.get('half', []), '2'),
                ('whole', detected.get('whole', []), '1'),
                ('sixteenth', detected.get('sixteenth', []), '0.5')  # Add sixteenth notes support
            ]:
                for box in boxes:
                    if in_staff(box):
                        note = NoteEvt(box, duration, staff, sharps, flats)
                        staff_notes.append(note)
                        
                        # Draw on debug image with random color
                        color = (randint(0, 255), randint(0, 255), randint(0, 255))
                        box.draw(notes_debug, color)
            
            # Sort notes by x position (left to right)
            staff_notes.sort(key=lambda n: n.box.x)
            self.notes.append(staff_notes)
        
        self.debug_images["notes"] = notes_debug
        return self
    
    def _load_templates(self, tmpl_dir, file_list):
        """Load template images from the template directory"""
        templates = []
        for file_name in file_list:
            path = os.path.join(tmpl_dir, file_name)
            if os.path.exists(path):
                template = cv2.imread(path, 0)
                if template is not None:
                    templates.append(template)
                else:
                    logging.warning(f"Failed to load template: {path}")
            else:
                logging.warning(f"Template file not found: {path}")
        
        return templates
    
    def _detect_symbols(self, img, templates, scale_range=(50, 150), threshold=0.7):
        """Detect symbols using template matching with scale search"""
        if not templates:
            return []
        
        h, w = img.shape
        best_matches = []
        best_scale = 1.0
        best_count = -1
        
        # Search at different scales
        for scale_pct in range(scale_range[0], scale_range[1]+1, 5):
            scale = scale_pct / 100.0
            current_matches = []
            match_count = 0
            
            for template in templates:
                try:
                    # Resize template
                    t_h, t_w = template.shape
                    resized = cv2.resize(template, None, fx=scale, fy=scale, 
                                        interpolation=cv2.INTER_CUBIC)
                    
                    # Match template
                    result = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= threshold)
                    points = list(zip(*locations[::-1]))
                    
                    # Create bounding boxes
                    rt_h, rt_w = resized.shape
                    boxes = [BBox(x, y, rt_w, rt_h) for x, y in points]
                    
                    current_matches.extend(boxes)
                    match_count += len(boxes)
                except Exception as e:
                    logging.warning(f"Template matching error: {e}")
            
            # Keep best scale
            if match_count > best_count:
                best_matches = current_matches
                best_count = match_count
                best_scale = scale
        
        # Merge overlapping matches
        merged = self._merge_regions(best_matches, 0.5)
        return merged
    
    def generate_midi(self, output_path):
        """Generate MIDI file from detected notes"""
        if not self.notes:
            logging.error("No notes detected, cannot generate MIDI")
            return False
        
        # Create MIDI file
        midi = MIDIFile(1)
        midi.addTrackName(0, 0, "Violin")
        midi.addTempo(0, 0, 120)
        
        # Add notes to MIDI file
        time_position = 0
        total_notes = 0
        
        for staff_notes in self.notes:
            for note in staff_notes:
                midi.addNote(0, 0, note.pitch, time_position, note.dur, 100)
                time_position += note.dur
                total_notes += 1
        
        # Write MIDI file
        with open(output_path, 'wb') as f:
            midi.writeFile(f)
        
        logging.info(f"Generated MIDI file: {output_path} with {total_notes} notes")
        return True
    
    def save_debug_images(self, prefix="debug_", output_dir="static"):
        """Save debug images to disk with input filename prefix."""
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]

        for name, img in self.debug_images.items():
            filename = f"{base_name}_{prefix}{name}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img)
            logging.info(f"Saved debug image: {filepath}")
        
        return self

def convert(img_path, out_mid, tmpl_dir='resources/template', debug=False, force_all=False):
    """Main conversion function: Sheet music to Violin MIDI"""
    try:
        # Create analyzer and process the image
        analyzer = ScoreAnalyzer(img_path, debug=debug)
        analyzer.preprocess(auto_crop=True)
        analyzer.detect_staffs()
        analyzer.detect_violin_parts() if not force_all else analyzer.detect_violin_parts_force_all()
        analyzer.detect_notes(tmpl_dir=tmpl_dir)
        
        # Generate MIDI file
        success = analyzer.generate_midi(out_mid)
        
        # Save debug images if requested
        if debug:
            analyzer.save_debug_images(output_dir=STATIC_FOLDER)
        
        if success:
            logging.info(f"✓ Successfully converted {img_path} to {out_mid}")
            return True
        else:
            logging.error(f"Failed to generate MIDI for {img_path}")
            return False
            
    except Exception as e:
        logging.error(f"Conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Universal Violin Sheet Music to MIDI Converter')
    parser.add_argument('image', help='Path to sheet music image')
    parser.add_argument('output', nargs='?', default='output.mid', help='Output MIDI file path')
    parser.add_argument('--tmpl', default='resources/template', help='Template directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualizations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--force-all', action='store_true', help='Use all detected staffs regardless of instrument labels')

    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert image to MIDI
    result = convert(args.image, args.output, tmpl_dir=args.tmpl, debug=args.debug)
    
    # Return appropriate exit code
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())