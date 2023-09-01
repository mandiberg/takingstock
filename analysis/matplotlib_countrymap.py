import folium
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from my_declarative_base import Location, Images
from mp_db_io import DataIO
from sqlalchemy.pool import NullPool

# Setup your credentials and database connection as before
io = DataIO()
db = io.db
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)
Session = sessionmaker(bind=engine)
session = Session()

# Retrieve data from the database
data = session.query(Location.nation_name_alpha, Location.code_alpha3, Location.code_iso,
                      Images.location_id).join(Images, Images.location_id == Location.location_id)\
                      .filter(Images.site_name_id == 4).limit(100000000)

# Create a map centered around the data
m = folium.Map(location=[0, 0], zoom_start=2)

# Process data to count occurrences of each country
country_counts = {}
for nation_name_alpha, code_alpha3, code_iso, _ in data:
    if code_alpha3:
        country_code = code_alpha3
    else:
        country_code = code_iso
    if country_code not in country_counts:
        country_counts[country_code] = 0
    country_counts[country_code] += 1

# Normalize the values for colormap
min_value = min(country_counts.values())
max_value = max(country_counts.values())

# Create a function to determine color based on occurrences
def get_color(value):
    normalized_value = (value - min_value) / (max_value - min_value)
    return '#{:02x}{:02x}{:02x}'.format(int(255 * normalized_value), 0, int(255 * (1 - normalized_value)))

# Plot the countries with color based on counts
for country_code, value in country_counts.items():
    print(country_code, str(value))
    x, y = 0, 0  # Placeholder coordinates since folium doesn't use geographic coordinates directly
    country_color = get_color(value)
    folium.CircleMarker(location=[y, x], radius=5, color=country_color, fill=True, fill_color=country_color).add_to(m)

# Display the map
m.save("country_occurrences_map.html")
