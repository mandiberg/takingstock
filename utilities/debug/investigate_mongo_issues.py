"""
Investigate MongoDB encoding issues for problematic image_ids.
Checks format, type, and structure of face_encodings68, body_landmarks, and hand_results.
"""

import pandas as pd
import json
import os
from pathlib import Path

# importing project-specific models
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)


from mp_db_io import DataIO

# Load the problematic image_ids
csv_path = '/Users/michael.mandiberg/Documents/GitHub/takingstock/utilities/debug/mongo_issues_imageids.csv'
try:
    df = pd.read_csv(csv_path, header=None, names=['image_id'])
    problem_ids = df['image_id'].tolist()
    print(f"Loaded {len(problem_ids)} problematic image_ids from CSV")
except FileNotFoundError:
    print(f"CSV not found at {csv_path}, using sample from error message")
    problem_ids = [125797492, 125798252, 125798559, 125799121, 125800777, 
                   125801666, 125802754, 125803219, 125803668, 125805063]

print(f"Investigating {len(problem_ids)} image_ids...\n")

# Initialize DataIO
io = DataIO()

# Track issue types
issues = {
    'missing_face_encodings68': [],
    'missing_body_landmarks': [],
    'missing_hand_results': [],
    'wrong_type_face_encodings68': [],
    'wrong_type_body_landmarks': [],
    'wrong_type_hand_results': [],
    'mediapipe_protobuf': [],
    'dict_format': [],
    'corrupted': [],
    'success': [],
}

def check_type(value, field_name):
    """Check the type and structure of a MongoDB field value."""
    if value is None:
        return 'None/Missing'
    
    value_type = type(value).__name__
    
    # Check for mediapipe protobuf objects
    if 'mediapipe' in str(type(value)).lower() or 'proto' in value_type.lower():
        return f'MediaPipe_Protobuf ({value_type})'
    
    # Check for dict
    if isinstance(value, dict):
        keys = list(value.keys())[:5]  # First 5 keys
        return f'Dict (keys: {keys})'
    
    # Check for list
    if isinstance(value, list):
        if len(value) > 0:
            first_elem_type = type(value[0]).__name__
            return f'List[{first_elem_type}] (len={len(value)})'
        return 'List (empty)'
    
    # Check for numpy array
    if hasattr(value, 'shape'):
        return f'NumPy_Array (shape={value.shape})'
    
    # Check for bytes/binary
    if isinstance(value, bytes):
        return f'Bytes (len={len(value)})'
    
    return f'Unknown ({value_type})'

# Investigate each image
for idx, image_id in enumerate(problem_ids[:50]):  # Limit to first 50 for speed
    if idx % 10 == 0:
        print(f"Processing {idx}/{len(problem_ids[:50])}...")
    
    try:
        # Fetch encoding data
        result = io.get_encodings_mongo(image_id)
        
        # Extract fields
        if isinstance(result, pd.Series):
            face_encodings68 = result.get('face_encodings68')
            face_landmarks = result.get('face_landmarks')
            body_landmarks = result.get('body_landmarks')
            body_landmarks_normalized = result.get('body_landmarks_normalized')
            body_landmarks_3D = result.get('body_landmarks_3D')
            hand_results = result.get('hand_results')
        else:
            face_encodings68, face_landmarks, body_landmarks, body_landmarks_normalized, body_landmarks_3D, hand_results = result
        
        # Check each field
        face_type = check_type(face_encodings68, 'face_encodings68')
        body_type = check_type(body_landmarks, 'body_landmarks')
        hand_type = check_type(hand_results, 'hand_results')
        
        # Categorize issues
        has_issue = False
        
        if face_encodings68 is None:
            issues['missing_face_encodings68'].append(image_id)
            has_issue = True
        elif 'MediaPipe_Protobuf' in face_type:
            issues['mediapipe_protobuf'].append((image_id, 'face_encodings68', face_type))
            has_issue = True
        elif 'Dict' in face_type and 'face_encodings68' in str(face_encodings68):
            # Old format might be wrapped in dict
            issues['dict_format'].append((image_id, 'face_encodings68', face_type))
            has_issue = True
        
        if body_landmarks is None:
            issues['missing_body_landmarks'].append(image_id)
            has_issue = True
        elif 'MediaPipe_Protobuf' in body_type:
            issues['mediapipe_protobuf'].append((image_id, 'body_landmarks', body_type))
            has_issue = True
        elif 'Dict' in body_type:
            # Check if it's the new dict format
            if isinstance(body_landmarks, dict) and 'landmark' in body_landmarks:
                issues['dict_format'].append((image_id, 'body_landmarks', body_type))
                has_issue = True
        
        if hand_results is None:
            issues['missing_hand_results'].append(image_id)
            has_issue = True
        elif 'MediaPipe_Protobuf' in hand_type:
            issues['mediapipe_protobuf'].append((image_id, 'hand_results', hand_type))
            has_issue = True
        elif isinstance(hand_results, dict) and ('left_hand' in hand_results or 'right_hand' in hand_results):
            # This is expected format, check deeper
            pass
        
        if not has_issue:
            issues['success'].append(image_id)
        
        # Print detailed info for first few problematic ones
        if has_issue and idx < 10:
            print(f"\n{'='*60}")
            print(f"Image ID: {image_id}")
            print(f"  face_encodings68: {face_type}")
            print(f"  body_landmarks: {body_type}")
            print(f"  hand_results: {hand_type}")
            
            # Try to show sample data
            if body_landmarks is not None and idx < 3:
                print(f"\n  Sample body_landmarks data:")
                if isinstance(body_landmarks, dict):
                    print(f"    Keys: {list(body_landmarks.keys())}")
                    if 'landmark' in body_landmarks:
                        print(f"    landmark type: {type(body_landmarks['landmark'])}")
                        if isinstance(body_landmarks['landmark'], list) and len(body_landmarks['landmark']) > 0:
                            print(f"    First landmark: {body_landmarks['landmark'][0]}")
                elif isinstance(body_landmarks, list) and len(body_landmarks) > 0:
                    print(f"    First element type: {type(body_landmarks[0])}")
                    print(f"    First element: {body_landmarks[0]}")
        
    except KeyError as e:
        print(f"\nKeyError for image {image_id}: {e}")
        issues['corrupted'].append((image_id, str(e)))
    except Exception as e:
        print(f"\nUnexpected error for image {image_id}: {type(e).__name__}: {e}")
        issues['corrupted'].append((image_id, str(e)))

# Print summary
print("\n" + "="*60)
print("ISSUE SUMMARY")
print("="*60)
print(f"Total images investigated: {len(problem_ids[:50])}")
print(f"\nMissing data:")
print(f"  face_encodings68: {len(issues['missing_face_encodings68'])} images")
print(f"  body_landmarks: {len(issues['missing_body_landmarks'])} images")
print(f"  hand_results: {len(issues['missing_hand_results'])} images")
print(f"\nFormat issues:")
print(f"  MediaPipe protobuf format: {len(issues['mediapipe_protobuf'])} occurrences")
print(f"  Dict format (new MediaPipe): {len(issues['dict_format'])} occurrences")
print(f"  Corrupted/Error: {len(issues['corrupted'])} images")
print(f"\nSuccess (no obvious issues): {len(issues['success'])} images")

# Show examples of each issue type
if issues['mediapipe_protobuf']:
    print(f"\n{'='*60}")
    print("MEDIAPIPE PROTOBUF EXAMPLES (first 5):")
    for item in issues['mediapipe_protobuf'][:5]:
        print(f"  Image {item[0]}, field '{item[1]}': {item[2]}")

if issues['dict_format']:
    print(f"\n{'='*60}")
    print("DICT FORMAT EXAMPLES (first 5):")
    for item in issues['dict_format'][:5]:
        print(f"  Image {item[0]}, field '{item[1]}': {item[2]}")

if issues['corrupted']:
    print(f"\n{'='*60}")
    print("CORRUPTED/ERROR EXAMPLES (first 5):")
    for item in issues['corrupted'][:5]:
        print(f"  Image {item[0]}: {item[1]}")

# Save detailed report
report_path = '/Users/michael.mandiberg/Documents/GitHub/takingstock/utilities/debug/mongo_issues_report.txt'
with open(report_path, 'w') as f:
    f.write("MongoDB Encoding Issues Investigation\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Total images investigated: {len(problem_ids[:50])}\n\n")
    
    f.write("Missing face_encodings68:\n")
    for img_id in issues['missing_face_encodings68']:
        f.write(f"  {img_id}\n")
    
    f.write(f"\nMissing body_landmarks:\n")
    for img_id in issues['missing_body_landmarks']:
        f.write(f"  {img_id}\n")
    
    f.write(f"\nMediaPipe protobuf format:\n")
    for item in issues['mediapipe_protobuf']:
        f.write(f"  Image {item[0]}, field '{item[1]}': {item[2]}\n")
    
    f.write(f"\nDict format (new MediaPipe):\n")
    for item in issues['dict_format']:
        f.write(f"  Image {item[0]}, field '{item[1]}': {item[2]}\n")
    
    f.write(f"\nCorrupted/Error:\n")
    for item in issues['corrupted']:
        f.write(f"  Image {item[0]}: {item[1]}\n")

print(f"\n{'='*60}")
print(f"Detailed report saved to: {report_path}")
