import json
import random

# Base map template
base_map = [
    "BBBBBBBBBBBBBBBBBBBB",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "B..................B",
    "BBBBBBBBBBBBBBBBBBBB"
]


def generate_map(difficulty):
    # Create a copy of the base map
    map_data = [list(row) for row in base_map]

    # Determine number of enemies based on difficulty
    if difficulty <= 3:
        num_enemies = random.randint(1, 3)
    elif difficulty <= 6:
        num_enemies = random.randint(4, 6)
    else:
        num_enemies = random.randint(7, 10)

    # Place player (P)
    player_placed = False
    while not player_placed:
        x = random.randint(1, 18)
        y = random.randint(1, 13)
        if map_data[y][x] == '.':
            map_data[y][x] = 'P'
            player_placed = True

    # Place enemies (E)
    enemies_placed = 0
    while enemies_placed < num_enemies:
        x = random.randint(1, 18)
        y = random.randint(1, 13)
        if map_data[y][x] == '.':
            map_data[y][x] = 'E'
            enemies_placed += 1

    # Add blocks (B) based on difficulty
    if difficulty >= 2:
        # Simple obstacles for lower difficulties
        num_blocks = difficulty * 3
        blocks_placed = 0
        while blocks_placed < num_blocks:
            x = random.randint(1, 18)
            y = random.randint(1, 13)
            if map_data[y][x] == '.':
                map_data[y][x] = 'B'
                blocks_placed += 1

    # Add more complex structures for higher difficulties
    if difficulty >= 5:
        # Add some wall segments
        for _ in range(difficulty - 3):
            # Horizontal wall
            if random.random() > 0.5:      # probability to create a horizontal wall is 50%
                y = random.randint(3, 11)
                start_x = random.randint(2, 10)
                length = random.randint(3, 6)
                for x in range(start_x, min(19, start_x + length)):
                    if map_data[y][x] == '.':
                        map_data[y][x] = 'B'
            # Vertical wall
            else:
                x = random.randint(3, 16)
                start_y = random.randint(2, 8)
                length = random.randint(3, 6)
                for y in range(start_y, min(14, start_y + length)):
                    if map_data[y][x] == '.':
                        map_data[y][x] = 'B'

    # Convert back to strings
    map_strings = [''.join(row) for row in map_data]
    return '\n'.join(map_strings)


# Generate 1000 samples
samples = []
for i in range(10): #Previous value 1000
    # Vary difficulty between 1-10
    difficulty = (i % 10) + 1 # Previous value 10
    if i >= 9:  # Last 100 samples are extra challenging # Previous value 900
        difficulty = random.randint(8, 10)

    map_str = generate_map(difficulty)
    samples.append({
        "difficulty": difficulty,
        "map": map_str
    })


# Save to JSON file
with open('map_dataset.json', 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\n')

print("Dataset generated with 5000 samples!")