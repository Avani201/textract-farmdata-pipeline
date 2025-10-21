# AWS Lambda (Python) for OCR + parsing of 1860 agricultural census pages.
# Uses Textract + S3, merges continuation lines, outputs standardized CSVs.
# Written for RA project digitizing historical farm data.

import os
import csv
import io
import time
import urllib.parse
import re
import logging
import boto3




logger = logging.getLogger()
logger.setLevel(logging.INFO)
S3 = boto3.client("s3")




BUCKET = os.environ.get("BUCKET", "westvirginia-farmdata")
RAW_PREFIX = os.environ.get("RAW_PREFIX", "rawfarmdata/")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "processedfarmdata/")
ADD_NOTES_COLUMN = os.environ.get("ADD_NOTES_COLUMN", "1") == "1"




def get_bucket_region(bucket):
    resp = S3.get_bucket_location(Bucket=bucket)
    loc = resp.get("LocationConstraint")
    return loc or "us-east-1"




def textract_client_for_bucket(bucket):
    region = get_bucket_region(bucket)
    return boto3.client("textract", region_name=region)




def start_textract_ocr(bucket, key):
    textract = textract_client_for_bucket(bucket)
    resp = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    return resp["JobId"], textract




def wait_for_job(textract, job_id, delay=2, max_wait=900):
    waited = 0
    while True:
        r = textract.get_document_text_detection(JobId=job_id)
        st = r["JobStatus"]
        if st in ("SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"):
            return r
        time.sleep(delay)
        waited += delay
        if waited >= max_wait:
            raise TimeoutError(f"Textract job {job_id} timed out")




def fetch_all_blocks(textract, first_page, job_id):
    blocks = first_page.get("Blocks", [])
    token = first_page.get("NextToken")
    while token:
        r = textract.get_document_text_detection(JobId=job_id, NextToken=token)
        blocks.extend(r.get("Blocks", []))
        token = r.get("NextToken")
    return blocks




def get_bounding_box_info(block):
    bbox = block.get("Geometry", {}).get("BoundingBox", {})
    return {
        'left': bbox.get('Left', 0),
        'top': bbox.get('Top', 0),
        'width': bbox.get('Width', 0),
        'height': bbox.get('Height', 0)
    }




def analyze_column_structure(blocks):
    line_positions = []
    for block in blocks:
        if block.get("BlockType") == "LINE":
            bbox_info = get_bounding_box_info(block)
            text = block.get("Text", "").strip()
            if text and not is_header_text(text):
                line_positions.append(bbox_info['left'])
   
    if len(line_positions) < 10:
        return 0.5
   
    line_positions.sort()
    min_pos = min(line_positions)
    max_pos = max(line_positions)
    range_width = max_pos - min_pos
   
    if range_width < 0.1:
        return 0.5
   
    bin_size = range_width / 20
    bins = [0] * 20
   
    for pos in line_positions:
        bin_index = min(19, int((pos - min_pos) / bin_size))
        bins[bin_index] += 1
   
    min_count = float('inf')
    gap_bin = 10
    for i in range(5, 15):
        if bins[i] < min_count:
            min_count = bins[i]
            gap_bin = i
   
    threshold = min_pos + (gap_bin + 0.5) * bin_size
    return threshold




def is_header_text(text):
    """Return True only for actual document headers, never for numeric data."""
    if not text:
        return False
    text_lower = text.lower().strip()
   
    # Only filter actual document headers - never numeric data
    header_phrases = [
        'wayne county', 'west virginia', 'agricultural census', '1860',
        'name of owner', 'acres of improved', 'acres of unimproved',
        'cash value', 'value of farming', 'value of livestock',
        'columns 1, 2, 3', 'university of north carolina',
        'some parts of this county', 'the university of',
        'filmed the 1860', 'represent the following',
        'pages were filmed out of sequence', 'they were transcribed in the order',
        'in which they were filmed', 'fayette county', 'floyd county virginia',
        'hardy county'  # Added for Hardy County
    ]
   
    # Only return True if text contains a header phrase
    return any(phrase in text_lower for phrase in header_phrases)




def is_numeric_token(token):
    if not token:
        return False
    # Treat standalone dashes as valid numeric placeholders
    if token.strip() in ['-', '—', '–']:
        return True
    cleaned = token.replace(',', '').replace('$', '').replace('-', '').strip()
    if re.fullmatch(r'\d+(\.\d+)?', cleaned):
        return True
    if re.fullmatch(r'\$?\d[\d,]*([.]\d+)?', token.replace('-', '')):
        return True
    return False




def starts_with_name(text):
    if not text:
        return False
    head = text.split(',', 1)[0].strip()
    if not head:
        return False
   
    tokens = head.split()
    if not tokens:
        return False
   
    # Skip leading junk tokens (underscores, dashes, etc.)
    start_index = 0
    for i, token in enumerate(tokens):
        if re.match(r'^[_\-]+$', token):  # Skip tokens that are just underscores or dashes
            start_index = i + 1
        else:
            break
   
    # If all tokens were junk, not a name
    if start_index >= len(tokens):
        return False
   
    # Look for pattern: initials or abbreviated names followed by a proper name
    for i in range(start_index, len(tokens)):
        token = tokens[i]
        if re.match(r'^[A-Z]\.?$', token):  # Initial like "A." or "A"
            continue
        elif re.match(r'^[A-Z][a-z]*\.?$', token):  # Proper name or abbreviation like "Stemple", "Robt.", "Wm."
            return True
        else:
            break
   
    # Fallback to original logic
    first_token = tokens[start_index]
    if not first_token[0].isupper():
        return False
    if re.fullmatch(r'\d+', first_token):
        return False
    if not re.search(r'[A-Za-z]', first_token):
        return False
    return True




def is_continuation_line(text):
    """
    FIXED: Better detection of continuation lines containing orphaned numbers.
    A continuation line should contain ONLY numeric data and separators, with NO name.
    """
    if not text or not text.strip():
        return False
   
    text = text.strip()
   
    # If the line starts with what looks like a name pattern, it's NOT a continuation
    if starts_with_name(text):
        return False
   
    # Check if line contains only numeric values, dashes, commas, spaces, and dollar signs
    # Remove all valid numeric content and separators
    cleaned = text
    cleaned = re.sub(r'\d+', '', cleaned)  # Remove all numbers
    cleaned = re.sub(r'[,\s\-—–$\.]', '', cleaned)  # Remove separators, spaces, dashes, dollar signs, periods
   
    # If almost nothing remains after removing numbers and separators, it's likely a continuation
    if len(cleaned.strip()) <= 1:  # Allow for one stray character
        return True
   
    # Additional check: if line has no alphabetic characters except single letters
    # (which might be stray OCR artifacts), treat as continuation
    alpha_chars = re.findall(r'[A-Za-z]', text)
    if len(alpha_chars) <= 2:  # Very few letters = likely just numbers with OCR noise
        return True
   
    return False




def parse_structured_line_with_positions(text):
    """
    FIXED: Keep dashes as dashes during processing, only convert to "0" at CSV output.
    Everything after the name is considered numeric data in positional order.
    """
    # Clean up the text
    text = re.sub(r'\s+', ' ', text.strip())
    parts = [p.strip() for p in text.split(',')]
   
    # Find where the name ends - look for first numeric value or dash
    name_parts = []
    numeric_start_index = None
   
    for i, part in enumerate(parts):
        part = part.strip()
       
        # If this part contains a dash or number, this is where numeric data starts
        if (part in ['-', '—', '–', ''] or
            is_numeric_token(part) or
            re.search(r'\d', part)):
            numeric_start_index = i
            break
        else:
            # Check if this part has mixed name + numeric content
            tokens = part.split()
            found_numeric = False
            for j, token in enumerate(tokens):
                if (token in ['-', '—', '–'] or
                    is_numeric_token(token) or
                    re.search(r'\d', token)):
                    # Everything before this token is name
                    name_parts.extend(tokens[:j])
                    numeric_start_index = i
                    found_numeric = True
                    break
           
            if found_numeric:
                break
            else:
                name_parts.append(part)
   
    name = ' '.join(name_parts).strip()
   
    # Initialize result with dashes (not zeros)
    result = ['-'] * 5  # Keep as dashes during processing
   
    # Process numeric section positionally
    if numeric_start_index is not None:
        position_index = 0
       
        for i in range(numeric_start_index, len(parts)):
            if position_index >= 5:  # Don't exceed 5 columns
                break
               
            part = parts[i].strip()
           
            # Keep dashes as dashes
            if part in ['-', '—', '–', '']:
                result[position_index] = '-'
                position_index += 1
           
            # Process numeric values
            elif is_numeric_token(part) and part not in ['-', '—', '–']:
                cleaned = part.replace(',', '').replace('$', '').strip()
                if cleaned.isdigit():
                    result[position_index] = cleaned
                else:
                    result[position_index] = '-'  # Invalid number becomes dash
                position_index += 1
           
            # Handle mixed content in a single part
            else:
                tokens = part.split()
                for token in tokens:
                    if position_index >= 5:
                        break
                   
                    if token in ['-', '—', '–']:
                        result[position_index] = '-'
                        position_index += 1
                    elif is_numeric_token(token) and token not in ['-', '—', '–']:
                        cleaned = token.replace(',', '').replace('$', '').strip()
                        if cleaned.isdigit():
                            result[position_index] = cleaned
                        else:
                            result[position_index] = '-'
                        position_index += 1
                    # Skip any non-numeric, non-dash tokens
   
    # Special handling for single numeric values (preserve existing smart logic)
    numeric_values = [x for x in result if x != '-' and x.isdigit()]
    if len(numeric_values) == 1:
        single_value = int(numeric_values[0])
        comma_count = text.count(',')
       
        # Reset to all dashes, then apply smart assignment
        result = ['-'] * 5
       
        if comma_count == 1:
            result[4] = str(single_value)  # Livestock
        elif comma_count >= 4:
            result[4] = str(single_value)  # Livestock  
        elif single_value >= 10000:
            result[2] = str(single_value)  # Cash
        elif single_value >= 1000:
            result[2] = str(single_value)  # Cash
        elif single_value < 100:
            result[4] = str(single_value)  # Livestock
        else:
            result[4] = str(single_value)  # Default to Livestock
   
    return name, result




def extract_continuation_numbers_with_positions(text):
    """
    FIXED: Extract continuation data preserving dashes and maintaining positional structure.
    This function now preserves dashes as placeholders and maintains column positions.
    """
    if not text or not text.strip():
        return []
   
    text = text.strip()
   
    # Handle simple numeric-only lines first (like "450")
    if re.match(r'^\s*\d+\s*$', text):
        return [text.strip()]
   
    # Split by commas to preserve column structure
    parts = [part.strip() for part in text.split(',')]
   
    position_data = []
    for part in parts:
        part = part.strip()
       
        # Preserve dashes as placeholders
        if part in ['-', '—', '–', '']:
            position_data.append('-')
        # Handle numeric values (including those with $ signs)
        elif is_numeric_token(part) and part not in ['-', '—', '–']:
            cleaned = part.replace(',', '').replace('$', '').strip()
            if cleaned and cleaned.replace('.', '').isdigit():
                position_data.append(cleaned)
            else:
                position_data.append('-')
        else:
            # Try to extract numbers from mixed content, but if no clear number, use dash
            numbers = re.findall(r'\b\d{1,6}\b', part)
            if numbers:
                position_data.append(numbers[0])  # Take first number found
            else:
                # No valid number found, but we still need a placeholder
                position_data.append('-')
   
    return position_data




def merge_positional_data(base_data, continuation_data):
    """
    FIXED: Merge continuation data into base data, preserving dashes and exact positions.
    This fills empty positions (marked with '-') with continuation data in order.
    """
    if not continuation_data:
        return base_data
   
    # Ensure base_data has exactly 5 elements
    while len(base_data) < 5:
        base_data.append('-')
    base_data = base_data[:5]
   
    result = base_data[:]
   
    # Fill empty positions (marked with '-') with continuation data
    continuation_index = 0
    for i in range(5):
        # Only fill if current position is empty (dash) and we have continuation data
        if (result[i] == '-' and
            continuation_index < len(continuation_data)):
           
            # Get the next piece of continuation data
            cont_value = continuation_data[continuation_index]
           
            # Only use non-dash values to fill empty positions
            if cont_value != '-':
                result[i] = cont_value
           
            continuation_index += 1
   
    return result




def process_column_text_to_records(lines, column_name):
    """
    FIXED: Process records with enhanced continuation line handling.
    Better detection and merging of continuation lines.
    """
    records = []
    current_record = None
   
    logger.info(f"Processing {len(lines)} lines from {column_name}")
   
    for i, line in enumerate(lines):
        text = line['text'].strip()
        logger.debug(f"{column_name} Line {i+1}: '{text}'")
       
        # Special logging for Seymour Grady case
        if 'seymour grady' in text.lower():
            logger.info(f"SEYMOUR GRADY detected in line {i+1}: '{text}'")
       
        # Check if this is a continuation line (only numbers/dashes, no name)
        if is_continuation_line(text):
            logger.debug(f"Detected continuation line: '{text}'")
           
            if current_record:
                continuation_numbers = extract_continuation_numbers_with_positions(text)
               
                if continuation_numbers:
                    logger.info(f"Found continuation numbers: {continuation_numbers} for {current_record['name']}")
                   
                    # Log the state before merging
                    logger.info(f"Before merge - {current_record['name']}: {current_record['numbers']}")
                   
                    # Merge the continuation data with existing data
                    current_record['numbers'] = merge_positional_data(
                        current_record['numbers'],
                        continuation_numbers
                    )
                   
                    logger.info(f"After merge - {current_record['name']}: {current_record['numbers']}")
            else:
                logger.warning(f"Found continuation line '{text}' with no active record - skipping")
       
        elif starts_with_name(text):
            # This line starts with a name - it's a new record
            # Save previous record if it exists and has a name
            if current_record and current_record.get('name'):
                # Ensure exactly 5 values
                while len(current_record['numbers']) < 5:
                    current_record['numbers'].append("-")
                current_record['numbers'] = current_record['numbers'][:5]
                records.append(current_record)
               
                # Special logging for Seymour Grady
                if 'seymour grady' in current_record['name'].lower():
                    logger.info(f"SAVED SEYMOUR GRADY: {current_record['name']} -> {current_record['numbers']}")
           
            # Parse this line for structured data with position awareness
            name, position_data = parse_structured_line_with_positions(text)
           
            current_record = {
                'name': name,
                'numbers': position_data,  # Use the position data directly
                'page': line['page'],
                'page_line': line['page_line']
            }
           
            logger.debug(f"Started new record: {name}, numbers: {position_data}")
           
            # Special logging for Seymour Grady
            if 'seymour grady' in name.lower():
                logger.info(f"CREATED SEYMOUR GRADY: {name} -> {position_data} from '{text}'")
           
        else:
            # Line doesn't start with name and isn't a clear continuation
            # This might be a malformed line - try to extract any numeric data
            if current_record:
                # Try to extract numbers from this line as potential continuation
                potential_numbers = extract_continuation_numbers_with_positions(text)
               
                if potential_numbers and any(x != '-' for x in potential_numbers):
                    logger.debug(f"Found potential continuation in mixed line: {potential_numbers}")
                   
                    current_record['numbers'] = merge_positional_data(
                        current_record['numbers'],
                        potential_numbers
                    )
                   
                    logger.debug(f"After mixed line merge: {current_record['name']} -> {current_record['numbers']}")
                else:
                    logger.debug(f"Skipping line with no useful data: '{text}'")
            else:
                logger.debug(f"Skipping orphaned line: '{text}'")
   
    # Don't forget the last record
    if current_record and current_record.get('name'):
        # Ensure exactly 5 values
        while len(current_record['numbers']) < 5:
            current_record['numbers'].append("-")
        current_record['numbers'] = current_record['numbers'][:5]
        records.append(current_record)
       
        # Special logging for Seymour Grady
        if 'seymour grady' in current_record['name'].lower():
            logger.info(f"FINAL SEYMOUR GRADY: {current_record['name']} -> {current_record['numbers']}")
   
    logger.info(f"Extracted {len(records)} person records from {column_name}")
   
    # Show first few records for debugging
    for i, record in enumerate(records[:5]):
        logger.info(f"{column_name} Record {i+1}: {record['name']} -> {record['numbers']}")
   
    return records




def textract_to_records(blocks):
    # Use fixed threshold of 0.5 for two-column layout
    threshold = 0.5
    all_records = []
   
    # Group all lines by page
    pages_data = {}
   
    for block in blocks:
        if block.get("BlockType") == "LINE":
            bbox_info = get_bounding_box_info(block)
            text = block.get("Text", "").strip()
           
            if text and not is_header_text(text):
                page = block.get('Page', 1)
               
                if page not in pages_data:
                    pages_data[page] = []
               
                line_data = {'text': text, 'page': page, 'top': bbox_info['top'], 'left': bbox_info['left']}
                pages_data[page].append(line_data)
   
    # Process each page: left column first, then right column
    for page_num in sorted(pages_data.keys()):
        logger.info(f"Processing page {page_num}")
       
        # Separate into left and right columns using fixed threshold
        left_lines = [line for line in pages_data[page_num] if line['left'] < threshold]
        right_lines = [line for line in pages_data[page_num] if line['left'] >= threshold]
       
        # Sort both columns by top position to ensure proper top-to-bottom order
        left_lines.sort(key=lambda x: x['top'])
        right_lines.sort(key=lambda x: x['top'])
       
        # Assign page line numbers for left column
        if left_lines:
            # Find all name lines in left column
            left_name_lines = [line for line in left_lines if starts_with_name(line['text'])]
           
            # Create mapping from top position to page line number for left column
            top_to_page_line = {}
            current_page_line = 1
           
            for i, name_line in enumerate(left_name_lines):
                if i == 0:
                    top_to_page_line[name_line['top']] = current_page_line
                else:
                    # Check if this name is on same horizontal band as previous
                    if abs(name_line['top'] - left_name_lines[i-1]['top']) > 0.01:
                        current_page_line += 1
                    top_to_page_line[name_line['top']] = current_page_line
           
            # Assign page line numbers to all left column lines
            for line in left_lines:
                if starts_with_name(line['text']):
                    line['page_line'] = top_to_page_line[line['top']]
                else:
                    line['page_line'] = 1  # Default for continuation lines
       
        # Assign page line numbers for right column
        if right_lines:
            # Find all name lines in right column
            right_name_lines = [line for line in right_lines if starts_with_name(line['text'])]
           
            # Create mapping from top position to page line number for right column
            top_to_page_line = {}
            current_page_line = 1
           
            for i, name_line in enumerate(right_name_lines):
                if i == 0:
                    top_to_page_line[name_line['top']] = current_page_line
                else:
                    # Check if this name is on same horizontal band as previous
                    if abs(name_line['top'] - right_name_lines[i-1]['top']) > 0.01:
                        current_page_line += 1
                    top_to_page_line[name_line['top']] = current_page_line
           
            # Assign page line numbers to all right column lines
            for line in right_lines:
                if starts_with_name(line['text']):
                    line['page_line'] = top_to_page_line[line['top']]
                else:
                    line['page_line'] = 1  # Default for continuation lines
       
        # Process LEFT column for this page FIRST
        if left_lines:
            logger.info(f"Page {page_num} LEFT column: {len(left_lines)} lines")
            left_records = process_column_text_to_records(left_lines, f"Page {page_num} LEFT")
            all_records.extend(left_records)
            logger.info(f"Added {len(left_records)} records from page {page_num} LEFT column")
       
        # Process RIGHT column for this page SECOND
        if right_lines:
            logger.info(f"Page {page_num} RIGHT column: {len(right_lines)} lines")
            right_records = process_column_text_to_records(right_lines, f"Page {page_num} RIGHT")
            all_records.extend(right_records)
            logger.info(f"Added {len(right_records)} records from page {page_num} RIGHT column")
   
    logger.info(f"Total records processed: {len(all_records)}")
    return all_records




def split_name(full_name):
    if not full_name:
        return "", "", ""
   
    # Remove parenthetical content for splitting
    name_clean = re.sub(r'\([^)]+\)', '', full_name).strip()
    parts = name_clean.split()
   
    if not parts:
        return "", "", ""
   
    if len(parts) == 1:
        return parts[0], "", ""
   
    # Common suffixes to recognize
    suffixes = ['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v', '2nd', '3rd', '4th', '5th']
   
    # Check if last part is a suffix
    suffix = ""
    working_parts = parts[:]
   
    if len(parts) > 1 and parts[-1].lower() in suffixes:
        suffix = parts[-1]
        working_parts = parts[:-1]
   
    if len(working_parts) == 1:
        return working_parts[0], "", suffix
   
    # Last remaining word is surname, everything else is given names
    surname = working_parts[-1]
    given_names = " ".join(working_parts[:-1])
   
    return surname, given_names, suffix




def extract_alternate_name(full_name):
    if not full_name:
        return ""
    match = re.search(r'\(([^)]+)\)', full_name)
    return match.group(1).strip() if match else ""




def normalize_to_csv_format(records):
    """
    FIXED: Keep dashes as dashes in CSV output - preserve exact original format.
    """
    headers = [
        "Name", "Alternate Name", "Surname", "Given Names", "Suffix",
        "Acres of Improved Land", "Acres of Unimproved Land",
        "Cash Value of the Farm", "Value of Farming Implements and Machinery",
        "Value of Livestock", "Page", "Page Line"
    ]
    if ADD_NOTES_COLUMN:
        headers.append("Notes")
   
    output_rows = [headers]
   
    for record in records:
        name = record['name'].strip()
        numbers = record['numbers']
        if not name:
            continue
       
        alternate_name = extract_alternate_name(name)
        surname, given_names, suffix = split_name(name)
       
        output_row = [name, alternate_name, surname, given_names, suffix]
       
        # Add exactly 5 numeric columns, keeping dashes as dashes
        for i in range(5):
            if i < len(numbers):
                value = str(numbers[i]).strip()
                # Keep dashes as dashes, only clean up empty/None values
                if value in ['', 'None']:
                    output_row.append("-")
                else:
                    output_row.append(value)  # Preserve dashes and numbers exactly
            else:
                output_row.append("-")  # Default empty positions to dash
               
        output_row.extend([record['page'], record['page_line']])
       
        if ADD_NOTES_COLUMN:
            output_row.append("")
        output_rows.append(output_row)
   
    return output_rows




def write_csv_to_s3(bucket, key, table_rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    for row in table_rows:
        w.writerow(row)
    S3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"))




def lambda_handler(event, context):
    records = event.get("Records", [])
    if not records:
        return {"ok": True, "message": "No Records in event."}


    for rec in records:
        s3info = rec.get("s3", {})
        bucket = (s3info.get("bucket", {}) or {}).get("name") or BUCKET
        raw_key = (s3info.get("object", {}) or {}).get("key", "")
        key = urllib.parse.unquote_plus(raw_key)


        if not key or not key.startswith(RAW_PREFIX):
            continue


        try:
            # Verify S3 object exists and is accessible
            S3.head_object(Bucket=bucket, Key=key)
           
            job_id, textract = start_textract_ocr(bucket, key)
            first = wait_for_job(textract, job_id)
            if first["JobStatus"] != "SUCCEEDED":
                raise RuntimeError(f"Textract failed with status: {first['JobStatus']}")
            blocks = fetch_all_blocks(textract, first, job_id)


            records = textract_to_records(blocks)
            csv_data = normalize_to_csv_format(records)


            base = key.split("/")[-1].rsplit(".", 1)[0] + ".csv"
            out_key = f"{OUT_PREFIX}{base}"
            write_csv_to_s3(bucket, out_key, csv_data)
           
        except Exception as e:
            logger.error(f"Error processing s3://{bucket}/{key}: {str(e)}")
            continue


    return {"ok": True}
