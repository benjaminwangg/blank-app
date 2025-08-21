import streamlit as st
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pathlib import Path
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Population Radius Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .coordinate-input {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def get_hexes_in_radius(center_lat, center_lon, radius_km, resolution=8):
    """Get all hexes within a given radius using apothem height"""
    # Get center hexagon
    center_hex = h3.latlng_to_cell(center_lat, center_lon, resolution)
    
    # Calculate number of rings needed using apothem height
    # At resolution 8, apothem height is 0.9204 km
    k = int(np.ceil(radius_km / 0.9204))
    
    # Get all hexagons within k rings
    hexagons = h3.grid_disk(center_hex, k)
    
    return hexagons, center_hex, k

@st.cache_data
def load_hex_data():
    """Load the hex data from parquet file with caching"""
    try:
        columns = ['h3', 'population', 'geometry', 'centroid', 'lat', 'lon', 'density_per_mi2']
        gdf = gpd.read_parquet(r"us_hexes_with_geonames.parquet", columns=columns)
        return gdf
    except Exception as e:
        st.error(f"Error loading hex data: {e}")
        return None

def find_hexes_and_population_for_coordinate(center_lat, center_lon, radius_km, hex_gdf, resolution=8):
    """Find hexes within radius and calculate population statistics for a single coordinate"""
    
    # Get hexes within radius
    hexagons, center_hex, k = get_hexes_in_radius(center_lat, center_lon, radius_km, resolution)
    
    # Filter to only hexagons in our radius that exist in our data
    radius_hexes = hex_gdf[hex_gdf['h3'].isin(hexagons)].copy()
    
    if radius_hexes.empty:
        return {
            'center_lat': center_lat,
            'center_lon': center_lon,
            'center_hex': str(center_hex),
            'radius_km': radius_km,
            'rings': k,
            'hexes_found': 0,
            'total_population': 0,
            'avg_population': 0,
            'total_area_km2': 0,
            'avg_distance_km': 0,
            'max_distance_km': 0
        }
    
    # Calculate distance from center for each hex
    radius_hexes['distance_from_center_km'] = radius_hexes.apply(
        lambda row: h3.great_circle_distance(
            (center_lat, center_lon), 
            (row['lat'], row['lon']), 
            unit='km'
        ), 
        axis=1
    )
    
    # Add area
    radius_hexes['area_km2'] = radius_hexes['h3'].apply(lambda h: h3.cell_area(h, unit='km^2'))
    
    # Calculate statistics
    total_pop = radius_hexes['population'].sum()
    avg_pop = radius_hexes['population'].mean()
    total_area = radius_hexes['area_km2'].sum()
    avg_distance = radius_hexes['distance_from_center_km'].mean()
    max_distance = radius_hexes['distance_from_center_km'].max()
    
    return {
        'center_lat': center_lat,
        'center_lon': center_lon,
        'center_hex': str(center_hex),
        'radius_km': radius_km,
        'rings': k,
        'hexes_found': len(radius_hexes),
        'total_population': total_pop,
        'avg_population': avg_pop,
        'total_area_km2': total_area,
        'avg_distance_km': avg_distance,
        'max_distance_km': max_distance
    }

def process_coordinates(coordinates, radius_km, resolution=8):
    """Process coordinates and return results"""
    hex_gdf = load_hex_data()
    if hex_gdf is None:
        return None, None
    
    results = []
    
    with st.spinner(f"Processing {len(coordinates)} coordinates..."):
        for i, (lat, lon, name) in enumerate(coordinates):
            # Progress bar
            progress = (i + 1) / len(coordinates)
            st.progress(progress)
            
            result = find_hexes_and_population_for_coordinate(lat, lon, radius_km, hex_gdf, resolution)
            result['location_name'] = name
            results.append(result)
    
    return results, hex_gdf

def create_download_link(df, filename, text):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Population Radius Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("Analyze population within specified radius around coordinates using H3 hexagons")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Parameters
    radius_km = st.sidebar.slider("Radius (km)", 1, 50, 5, help="Radius around each coordinate to analyze")
    resolution = st.sidebar.selectbox("H3 Resolution", [7, 8, 9, 10], index=1, 
                                    help="H3 resolution (8 = ~1km hexes, 9 = ~0.5km hexes)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**H3 Resolution Guide:**")
    st.sidebar.markdown("- 7: ~5km hexes")
    st.sidebar.markdown("- 8: ~1km hexes")
    st.sidebar.markdown("- 9: ~0.5km hexes")
    st.sidebar.markdown("- 10: ~0.1km hexes")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📍 Coordinate Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "CSV Upload", "Example Coordinates"],
            horizontal=True
        )
        
        coordinates = []
        
        if input_method == "Manual Entry":
            st.markdown('<div class="coordinate-input">', unsafe_allow_html=True)
            
            num_locations = st.number_input("Number of locations", min_value=1, max_value=20, value=3)
            
            for i in range(num_locations):
                st.subheader(f"Location {i+1}")
                col_lat, col_lon, col_name = st.columns([1, 1, 2])
                
                with col_lat:
                    lat = st.number_input(f"Latitude {i+1}", value=40.7580, format="%.4f", key=f"lat_{i}")
                with col_lon:
                    lon = st.number_input(f"Longitude {i+1}", value=-73.9855, format="%.4f", key=f"lon_{i}")
                with col_name:
                    name = st.text_input(f"Location Name {i+1}", value=f"Location {i+1}", key=f"name_{i}")
                
                coordinates.append((lat, lon, name))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif input_method == "CSV Upload":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    # Check required columns
                    required_cols = ['latitude', 'longitude', 'name']
                    if all(col in df.columns for col in required_cols):
                        coordinates = list(zip(df['latitude'], df['longitude'], df['name']))
                        st.success(f"✅ Loaded {len(coordinates)} coordinates from CSV")
                    else:
                        st.error("❌ CSV must have columns: latitude, longitude, name")
                        coordinates = []
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    coordinates = []
                    
        else:  # Example coordinates
            st.info("Using example coordinates for demonstration")
            coordinates = [
                (40.7580, -73.9855, "Times Square, NYC"),
                (40.7128, -74.0060, "Lower Manhattan, NYC"),
                (40.7505, -73.9934, "Penn Station, NYC"),
            ]
            st.write("Example coordinates loaded")
    
    with col2:
        st.header("📊 Quick Stats")
        
        if coordinates:
            st.metric("Locations", len(coordinates))
            st.metric("Radius", f"{radius_km} km")
            st.metric("H3 Resolution", f"Level {resolution}")
            
            # Show coordinates preview
            st.subheader("Coordinates Preview")
            coord_df = pd.DataFrame(coordinates, columns=['Latitude', 'Longitude', 'Name'])
            st.dataframe(coord_df, use_container_width=True)
        else:
            st.info("Add coordinates to see stats")
    
    # Analysis button
    if coordinates and st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.markdown("---")
        st.header("📈 Analysis Results")
        
        # Process coordinates
        results, hex_gdf = process_coordinates(coordinates, radius_km, resolution)
        
        if results:
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Population", f"{results_df['total_population'].sum():,}")
            with col2:
                st.metric("Total Hexes", f"{results_df['hexes_found'].sum():,}")
            with col3:
                st.metric("Avg Population/Location", f"{results_df['total_population'].mean():,.0f}")
            with col4:
                st.metric("Total Area", f"{results_df['total_area_km2'].sum():.1f} km²")
            
            # Results table
            st.subheader("Detailed Results")
            display_cols = ['location_name', 'hexes_found', 'total_population', 'total_area_km2', 'avg_distance_km']
            display_df = results_df[display_cols].copy()
            display_df.columns = ['Location', 'Hexes', 'Population', 'Area (km²)', 'Avg Distance (km)']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Population by Location")
                st.bar_chart(display_df.set_index('Location')['Population'])
            
            with col2:
                st.subheader("Area Coverage by Location")
                st.bar_chart(display_df.set_index('Location')['Area (km²)'])
            
            # Download options
            st.markdown("---")
            st.header("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download summary CSV
                csv_link = create_download_link(
                    results_df, 
                    f"population_analysis_{radius_km}km_radius.csv",
                    "📥 Download Summary CSV"
                )
                st.markdown(csv_link, unsafe_allow_html=True)
            
            with col2:
                # Download detailed results
                if st.button("📥 Download Detailed Hex Data"):
                    # Create zip file with individual location data
                    import zipfile
                    from io import BytesIO
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for result in results:
                            if result['hexes_found'] > 0:
                                # Get hexes for this location
                                center_hex = h3.string_to_h3(result['center_hex'])
                                hexagons, _, _ = get_hexes_in_radius(
                                    result['center_lat'], 
                                    result['center_lon'], 
                                    result['radius_km']
                                )
                                
                                # Filter hex data
                                location_hexes = hex_gdf[hex_gdf['h3'].isin(hexagons)].copy()
                                
                                # Add distance and area
                                location_hexes['distance_from_center_km'] = location_hexes.apply(
                                    lambda row: h3.great_circle_distance(
                                        (result['center_lat'], result['center_lon']), 
                                        (row['lat'], row['lon']), 
                                        unit='km'
                                    ), 
                                    axis=1
                                )
                                location_hexes['area_km2'] = location_hexes['h3'].apply(
                                    lambda h: h3.cell_area(h, unit='km^2')
                                )
                                
                                # Save to zip
                                safe_name = result['location_name'].replace(' ', '_').replace(',', '').replace('.', '')
                                filename = f"hexes_{safe_name}_{radius_km}km_radius.parquet"
                                
                                # Convert to bytes
                                parquet_buffer = BytesIO()
                                location_hexes.to_parquet(parquet_buffer)
                                parquet_buffer.seek(0)
                                
                                zip_file.writestr(filename, parquet_buffer.getvalue())
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="📦 Download ZIP with All Hex Data",
                        data=zip_buffer.getvalue(),
                        file_name=f"hex_data_{radius_km}km_radius.zip",
                        mime="application/zip"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Uses H3 for hexagonal grid analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
