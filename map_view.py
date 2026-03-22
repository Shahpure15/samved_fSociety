"""
SIMS — Solapur Intelligent Mobility SaaS
map_view.py: Interactive Folium map of Solapur's junction network.

Renders a live density map showing how congestion at the primary
feed junction (Saat Rasta) propagates to neighbouring junctions,
giving judges visual intuition for the Hive Mind signal coordination.
"""

from __future__ import annotations

import folium
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

# ------------------------------------------------------------------
# Real Solapur junction coordinates
# ------------------------------------------------------------------

JUNCTIONS: list[dict] = [
    {
        "name": "Saat Rasta",
        "lat": 17.6868,
        "lon": 75.9064,
        "camera_id": "CAM-001",
        "default_density": 88.0,
        "is_primary": True,
    },
    {
        "name": "Railway Station Rd",
        "lat": 17.6905,
        "lon": 75.9142,
        "camera_id": "CAM-047",
        "default_density": 72.0,
        "is_primary": False,
    },
    {
        "name": "Market Yard",
        "lat": 17.6798,
        "lon": 75.9020,
        "camera_id": "CAM-023",
        "default_density": 55.0,
        "is_primary": False,
    },
    {
        "name": "Hotgi Road Junction",
        "lat": 17.6850,
        "lon": 75.8990,
        "camera_id": "CAM-031",
        "default_density": 41.0,
        "is_primary": False,
    },
    {
        "name": "Hutatma Chowk",
        "lat": 17.6920,
        "lon": 75.9100,
        "camera_id": "CAM-058",
        "default_density": 28.0,
        "is_primary": False,
    },
    {
        "name": "Gold Chowk",
        "lat": 17.6830,
        "lon": 75.9155,
        "camera_id": "CAM-019",
        "default_density": 33.0,
        "is_primary": False,
    },
]

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _density_colour(density: float) -> str:
    """
    Returns a Folium-compatible colour string based on density level.

    > 80%  → red     (critical, Max-Pressure will extend green)
    50-80% → orange  (moderate congestion)
    < 50%  → green   (free flow)
    """
    if density >= 80:
        return "red"
    elif density >= 50:
        return "orange"
    else:
        return "green"


def _propagate_density(primary_density: float, junction_index: int) -> float:
    """
    Simulates how congestion at the primary junction (Saat Rasta)
    propagates to neighbouring junctions.

    In production, each junction would compute its own p_z(k).
    For the MVP demo, we simulate propagation: nearby junctions
    get 60-80% of primary density, further ones get 30-50%.

    Propagation factors per junction index (0=primary, skip it):
    index 1 (Railway Stn): 0.82  — very close, high propagation
    index 2 (Market Yard): 0.63  — moderate distance
    index 3 (Hotgi Rd):    0.47  — further away
    index 4 (Hutatma):     0.38  — far
    index 5 (Gold Chowk):  0.35  — far
    """
    factors = [1.0, 0.82, 0.63, 0.47, 0.38, 0.35]
    factor = factors[junction_index] if junction_index < len(factors) else 0.3
    return min(primary_density * factor, 100.0)


# ------------------------------------------------------------------
# Map builder
# ------------------------------------------------------------------

def build_junction_map(primary_density: float = 88.0) -> folium.Map:
    """
    Build a Folium map centred on Solapur showing all 6 junctions.

    primary_density: the live density % from the video feed (Saat Rasta).
    Returns a folium.Map ready to render with streamlit_folium.
    """
    m = folium.Map(
        location=[17.6868, 75.9064],
        zoom_start=14,
        tiles="CartoDB positron",  # clean light basemap, good for demo
    )

    primary_lat, primary_lon = 17.6868, 75.9064

    for i, j in enumerate(JUNCTIONS):
        # Compute this junction's density
        density = primary_density if j["is_primary"] else _propagate_density(primary_density, i)
        colour = _density_colour(density)

        if density >= 80:
            signal = "🟢 Extending Green +15s"
        elif density >= 50:
            signal = "🟡 Monitoring"
        else:
            signal = "✅ Normal Cycle"

        primary_badge = (
            '<br><b style="color:red">⚠ PRIMARY FEED — Live Analysis</b>'
            if j["is_primary"] else ""
        )

        popup_html = f"""
        <div style='font-family:sans-serif;width:200px'>
          <b style='font-size:14px'>{j["name"]}</b><br>
          <hr style='margin:4px 0'>
          Camera: {j["camera_id"]}<br>
          Density: <b style='color:{colour}'>{density:.0f}%</b><br>
          Signal: {signal}<br>
          Algorithm: Max-Pressure p_z(k)<br>
          {primary_badge}
        </div>
        """

        folium.CircleMarker(
            location=[j["lat"], j["lon"]],
            radius=18 if j["is_primary"] else 12,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f'{j["name"]}: {density:.0f}%',
        ).add_to(m)

        # Draw flow lines from Saat Rasta to each neighbour
        if not j["is_primary"]:
            mid_lat = (primary_lat + j["lat"]) / 2
            mid_lon = (primary_lon + j["lon"]) / 2

            folium.PolyLine(
                locations=[[primary_lat, primary_lon], [j["lat"], j["lon"]]],
                color=colour,
                weight=3,
                opacity=0.6,
                dash_array="8 6",
            ).add_to(m)

            # Arrow at midpoint showing flow direction
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:14px;color:{colour};">➤</div>',
                    icon_size=(18, 18),
                    icon_anchor=(9, 9),
                ),
            ).add_to(m)

    # Legend (bottom-right)
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; right: 10px;
        z-index: 1000;
        background: white;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #ccc;
        font-family: sans-serif;
        font-size: 12px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
    ">
        <b>Density Legend</b><br>
        🔴 Critical (&gt;80%) — Max-Pressure Active<br>
        🟠 Moderate (50-80%) — Monitoring<br>
        🟢 Free Flow (&lt;50%) — Normal Cycle
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ------------------------------------------------------------------
# Public Streamlit render function
# ------------------------------------------------------------------

def render_map(primary_density: float = 88.0) -> None:
    """
    Renders the Solapur junction network map in the current
    Streamlit context.

    Call this inside a Streamlit tab or container.
    primary_density: live density % from process_frame() metrics.
    """
    st.markdown("#### Solapur Junction Network — Live Density Map")
    st.caption(
        "Density propagates outward from the primary feed junction (Saat Rasta). "
        "Red = Max-Pressure active, extending green signal. "
        "Click any junction marker for details."
    )

    m = build_junction_map(primary_density)

    st_folium(
        m,
        width=None,
        height=500,
        returned_objects=[],
    )

    # Junction status table below the map
    st.markdown("##### Junction Status — This Frame")

    rows = []
    for i, j in enumerate(JUNCTIONS):
        d = primary_density if j["is_primary"] else _propagate_density(primary_density, i)
        status = "🔴 CRITICAL" if d >= 80 else ("🟠 MODERATE" if d >= 50 else "🟢 NORMAL")
        signal = "Extend Green +15s" if d >= 80 else ("Monitor" if d >= 50 else "Normal Cycle")
        rows.append({
            "Junction": j["name"],
            "Camera": j["camera_id"],
            "Density": f"{d:.0f}%",
            "Status": status,
            "Signal Decision": signal,
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )



