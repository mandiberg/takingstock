import random

def generate_random_numbers_with_specific_ids(start, end, number, specific_ids):
    """
    Generate a list of random numbers between start and end, ensuring specific IDs are included.
    
    Args:
        start (int): The lower bound for random numbers (inclusive)
        end (int): The upper bound for random numbers (inclusive)
        number (int): The total number of numbers to generate
        specific_ids (list): List of specific IDs that must be included
        
    Returns:
        list: A sorted list of random numbers including the specific IDs
    """
    # Validate inputs
    if number < len(specific_ids):
        raise ValueError("Number of requested numbers must be greater than or equal to the number of specific IDs")
    
    # Filter specific_ids to only include those within the range
    valid_specific_ids = [id for id in specific_ids if start <= id <= end]
    
    # Generate random numbers (excluding the specific IDs)
    remaining_count = number - len(valid_specific_ids)
    random_numbers = set()
    
    while len(random_numbers) < remaining_count:
        num = random.randint(start, end)
        if num not in valid_specific_ids and num not in random_numbers:
            random_numbers.add(num)
    
    # Combine and sort all numbers
    result = sorted(list(random_numbers) + valid_specific_ids)
    
    return result

# Set parameters
start = 0
end = 130000000
number = 50000

# Example specific image IDs to include (replace with your actual IDs)
# These are just placeholder IDs - replace these with your actual image IDs
specific_image_ids = [39617850, 33182898]

# Generate the random numbers including specific IDs
result = generate_random_numbers_with_specific_ids(start, end, number, specific_image_ids)

# Print the results
print(f"Generated {len(result)} random numbers between {start} and {end}")
print(f"Made sure to include these specific IDs: {specific_image_ids}")
print("\nResults:")
print(result)

# Verify specific IDs are included
for id in specific_image_ids:
    if id in result:
        print(f"Confirmed: ID {id} is included in the results")
    else:
        print(f"Error: ID {id} is not included in the results")