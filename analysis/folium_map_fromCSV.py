import folium
import pandas as pd

# Read the CSV file into a pandas DataFrame
data = pd.read_csv("_count_of_locations_SELECT_IFNULL_location_code_alpha3_location__202308191407.csv")

# Create a map centered around the data
m = folium.Map(location=[0, 0], zoom_start=2)

# Normalize the values for colormap
min_value = data['count'].min()
max_value = data['count'].max()

# Create a function to determine color based on occurrences
def get_color(value):
    normalized_value = (value - min_value) / (max_value - min_value)
    return '#{:02x}{:02x}{:02x}'.format(int(255 * normalized_value), 0, int(255 * (1 - normalized_value)))

# Plot the countries with color based on counts
for _, row in data.iterrows():
    x, y = 0, 0  # Placeholder coordinates since folium doesn't use geographic coordinates directly
    country_color = get_color(row['count'])
    folium.CircleMarker(location=[y, x], radius=5, color=country_color, fill=True, fill_color=country_color).add_to(m)

# Display the map
m.save("country_occurrences_map.html")
